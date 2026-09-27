import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from openlocalagent.model import LocalAgentLM, ModelConfig

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


def _make_bundle(tmp_path, *, action_only=False, tie_embeddings=True, fp16=True):
    """Export a small full bundle (model.onnx + heads.json + meta.json) and return paths + model."""
    from openlocalagent.agent.pointer_head import PointerHead
    from openlocalagent.agent.tool_head import ToolHead
    from openlocalagent.inference.export.to_onnx import export_web

    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=64, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=64, name="t",
                      tie_embeddings=tie_embeddings)
    m = LocalAgentLM(cfg).eval()
    th = ToolHead(cfg.d_model)
    ph = PointerHead(cfg.d_model)
    ckpt = tmp_path / "m.pt"
    torch.save({
        "cfg": cfg.__dict__,
        "state_dict": m.state_dict(),
        "tool_head": th.state_dict(),
        "ptr_head": ph.state_dict(),
    }, ckpt)
    out_dir = tmp_path / "web"
    stats = export_web(
        str(ckpt), str(out_dir), fp16=fp16, check=False, action_only=action_only
    )
    return stats, m, th, ph, cfg


def test_web_export_hidden_logits_parity(tmp_path):
    stats, m, th, ph, cfg = _make_bundle(tmp_path)
    sess = ort.InferenceSession(stats["model.onnx"], providers=["CPUExecutionProvider"])

    x = torch.randint(0, 256, (1, 13))
    with torch.no_grad():
        ref_logits, ref_hidden = m(x, return_hidden=True)
    ref_logits, ref_hidden = ref_logits.numpy(), ref_hidden.numpy()
    got_logits, got_hidden = sess.run(["logits", "hidden"], {"input_ids": x.numpy()})

    assert got_logits.shape == ref_logits.shape == (1, 13, 256)
    assert got_hidden.shape == ref_hidden.shape == (1, 13, cfg.d_model)
    assert np.abs(ref_logits - got_logits).max() < 1e-3
    assert np.abs(ref_hidden - got_hidden).max() < 1e-3


def test_action_only_export_matches_backbone_without_lm_head(tmp_path):
    stats, m, _th, _ph, cfg = _make_bundle(
        tmp_path, action_only=True, tie_embeddings=False
    )
    full_sess = ort.InferenceSession(stats["model.onnx"], providers=["CPUExecutionProvider"])
    action_sess = ort.InferenceSession(
        stats["action_model.onnx"], providers=["CPUExecutionProvider"]
    )
    action_fp16_sess = ort.InferenceSession(
        stats["action_model.fp16.onnx"], providers=["CPUExecutionProvider"]
    )

    x = torch.randint(0, cfg.vocab_size, (1, 13))
    with torch.no_grad():
        ref_hidden = m.forward_features(x).numpy()
    full_hidden = full_sess.run(["hidden"], {"input_ids": x.numpy()})[0]
    action_hidden = action_sess.run(["hidden"], {"input_ids": x.numpy()})[0]
    action_fp16_hidden = action_fp16_sess.run(["hidden"], {"input_ids": x.numpy()})[0]

    assert action_hidden.shape == ref_hidden.shape == (1, 13, cfg.d_model)
    assert np.abs(ref_hidden - action_hidden).max() < 1e-3
    assert np.abs(full_hidden - action_hidden).max() < 1e-3
    assert np.abs(ref_hidden - action_fp16_hidden).max() < 5e-2

    action_graph = onnx.load(stats["action_model.onnx"]).graph
    assert [output.name for output in action_graph.output] == ["hidden"]
    assert not any("lm_head" in initializer.name for initializer in action_graph.initializer)
    assert stats["action_model.onnx_MB"] < stats["model.onnx_MB"]

    meta = json.loads(open(stats["meta.json"]).read())
    assert meta["encoding"] == "utf-8-bytes"
    assert "tokenizer_file" not in meta
    assert meta["model_file"] == "model.fp16.onnx"
    assert meta["action_model_file"] == "action_model.fp16.onnx"
    assert meta["max_seq_len"] == cfg.max_seq_len
    assert meta["model_parameters"] == m.num_params()
    manifest = json.loads(open(stats["bundle-manifest.json"]).read())
    assert manifest["schema_version"] == 3
    assert manifest["parity_gate"]["hard_gate"] is True
    assert manifest["parity_gate"]["passed"] is True
    assert "model.fp16.onnx" in manifest["artifacts"]
    assert "action_model.fp16.onnx" in manifest["artifacts"]
    action_parity = manifest["parity_gate"]["results"]["action_model.fp16.onnx"]
    assert action_parity["passed"] is True
    assert action_parity["expected_outputs"] == ["hidden"]
    assert action_parity["precision"] == "fp16"
    assert action_parity["threshold_max_abs_diff"] == pytest.approx(5e-2)
    assert action_parity["max_abs_diff_by_output"]["hidden"] <= 5e-2
    assert action_parity["artifact"] == {
        "bytes": manifest["artifacts"]["action_model.fp16.onnx"]["bytes"],
        "sha256": manifest["artifacts"]["action_model.fp16.onnx"]["sha256"],
    }


def test_web_export_withholds_manifest_when_hard_parity_fails(tmp_path, monkeypatch):
    from openlocalagent.agent.pointer_head import PointerHead
    from openlocalagent.agent.tool_head import ToolHead
    from openlocalagent.inference.export import to_onnx

    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        embed_dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
        name="parity-failure",
    )
    model = LocalAgentLM(cfg).eval()
    checkpoint = tmp_path / "m.pt"
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state_dict": model.state_dict(),
            "tool_head": ToolHead(cfg.d_model).state_dict(),
            "ptr_head": PointerHead(cfg.d_model).state_dict(),
        },
        checkpoint,
    )
    out_dir = tmp_path / "web"
    out_dir.mkdir()
    manifest_path = out_dir / "bundle-manifest.json"
    manifest_path.write_text('{"stale":true}\n')

    def fail_parity(*_args, **_kwargs):
        raise RuntimeError("injected graph parity failure")

    monkeypatch.setattr(to_onnx, "_web_graph_parity", fail_parity)
    with pytest.raises(RuntimeError, match="injected graph parity failure"):
        to_onnx.export_web(
            str(checkpoint),
            str(out_dir),
            fp16=False,
            check=False,
            action_only=True,
        )
    assert not manifest_path.exists()


def _save_small_action_checkpoint(tmp_path):
    from openlocalagent.agent.pointer_head import PointerHead
    from openlocalagent.agent.tool_head import ToolHead

    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        embed_dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
        name="atomic-parity",
    )
    model = LocalAgentLM(cfg).eval()
    checkpoint = tmp_path / "atomic.pt"
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state_dict": model.state_dict(),
            "tool_head": ToolHead(cfg.d_model).state_dict(),
            "ptr_head": PointerHead(cfg.d_model).state_dict(),
        },
        checkpoint,
    )
    return checkpoint


def test_web_export_detects_graph_mutation_after_parity(tmp_path, monkeypatch):
    from openlocalagent.inference.export import to_onnx

    checkpoint = _save_small_action_checkpoint(tmp_path)
    out_dir = tmp_path / "web"
    real_gate = to_onnx._web_parity_gate

    def mutate_after_parity(model, graphs, cfg):
        evidence = real_gate(model, graphs, cfg)
        action_path = graphs["action_model.onnx"][0]
        assert action_path is not None
        with open(action_path, "ab") as handle:
            handle.write(b"concurrent mutation")
        return evidence

    monkeypatch.setattr(to_onnx, "_web_parity_gate", mutate_after_parity)
    with pytest.raises(RuntimeError, match="changed after parity"):
        to_onnx.export_web(
            str(checkpoint),
            str(out_dir),
            fp16=False,
            check=False,
            action_only=True,
        )
    assert not (out_dir / "bundle-manifest.json").exists()


def test_web_export_rejects_missing_declared_graph_before_manifest(tmp_path, monkeypatch):
    from openlocalagent.inference.export import to_onnx

    checkpoint = _save_small_action_checkpoint(tmp_path)
    out_dir = tmp_path / "web"
    real_gate = to_onnx._web_parity_gate

    def remove_before_parity(model, graphs, cfg):
        action_path = graphs["action_model.onnx"][0]
        assert action_path is not None
        Path(action_path).unlink()
        return real_gate(model, graphs, cfg)

    monkeypatch.setattr(to_onnx, "_web_parity_gate", remove_before_parity)
    with pytest.raises(FileNotFoundError, match="declared graph artifact action_model.onnx"):
        to_onnx.export_web(
            str(checkpoint),
            str(out_dir),
            fp16=False,
            check=False,
            action_only=True,
        )
    assert not (out_dir / "bundle-manifest.json").exists()


def test_bundle_manifest_rechecks_all_artifacts_before_atomic_publish(tmp_path, monkeypatch):
    from openlocalagent.inference.export import to_onnx

    graph = tmp_path / "model.onnx"
    heads = tmp_path / "heads.json"
    graph.write_bytes(b"parity-tested graph")
    heads.write_bytes(b'{"heads":"checked"}')
    graph_identity = to_onnx._bundle_artifact_identity(str(graph))
    real_identity = to_onnx._bundle_artifact_identity
    calls = {}

    def mutate_heads_before_final_check(path):
        calls[path] = calls.get(path, 0) + 1
        if path == str(heads) and calls[path] == 2:
            with open(path, "ab") as handle:
                handle.write(b" concurrent mutation")
        return real_identity(path)

    monkeypatch.setattr(to_onnx, "_bundle_artifact_identity", mutate_heads_before_final_check)
    with pytest.raises(RuntimeError, match="changed before manifest publication"):
        to_onnx._write_bundle_manifest(
            str(tmp_path),
            config_name="atomic",
            model_parameters=1,
            checkpoint_sha256="a" * 64,
            checkpoint_stage=None,
            checkpoint_step=None,
            model_config_sha256="b" * 64,
            artifacts={
                "heads.json": str(heads),
                "model.onnx": str(graph),
            },
            parity_gate={
                "hard_gate": True,
                "passed": True,
                "results": {
                    "model.onnx": {
                        "artifact": graph_identity,
                        "passed": True,
                    },
                },
            },
        )
    assert not (tmp_path / "bundle-manifest.json").exists()
    assert not (tmp_path / "bundle-manifest.json.tmp").exists()


def test_bundle_manifest_rejects_declared_graph_without_parity_result(tmp_path):
    from openlocalagent.inference.export import to_onnx

    model_graph = tmp_path / "model.onnx"
    action_graph = tmp_path / "action_model.onnx"
    model_graph.write_bytes(b"checked model")
    action_graph.write_bytes(b"unchecked action")
    with pytest.raises(RuntimeError, match="cover every declared ONNX graph exactly"):
        to_onnx._write_bundle_manifest(
            str(tmp_path),
            config_name="missing-parity",
            model_parameters=1,
            checkpoint_sha256="a" * 64,
            checkpoint_stage=None,
            checkpoint_step=None,
            model_config_sha256="b" * 64,
            artifacts={
                "action_model.onnx": str(action_graph),
                "model.onnx": str(model_graph),
            },
            parity_gate={
                "hard_gate": True,
                "passed": True,
                "results": {
                    "model.onnx": {
                        "artifact": to_onnx._bundle_artifact_identity(str(model_graph)),
                        "passed": True,
                    },
                },
            },
        )
    assert not (tmp_path / "bundle-manifest.json").exists()


def test_web_export_rejects_non_byte_vocab_without_tokenizer(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_web

    cfg = ModelConfig(vocab_size=320, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      ffn_hidden=128, max_seq_len=64, name="bpe")
    model = LocalAgentLM(cfg)
    checkpoint = tmp_path / "bpe.pt"
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, checkpoint)
    out_dir = tmp_path / "web"

    with pytest.raises(ValueError, match="requires tokenizer_path"):
        export_web(str(checkpoint), str(out_dir), fp16=False, check=False, action_only=True)
    assert not (out_dir / "model.onnx").exists()
    assert not (out_dir / "meta.json").exists()


def _make_bpe(tmp_path):
    from openlocalagent.model.tokenizer import train_bpe

    corpus = [
        "Open src/app.py and summarize the implementation.",
        "Search for the browser action handler, then run focused tests.",
        "<|user|>move report.md to archive/report.md<|assistant|>",
    ] * 20
    tokenizer_path = tmp_path / "source-tokenizer.json"
    tokenizer = train_bpe(corpus, tokenizer_path, vocab_size=320, min_frequency=1)
    return tokenizer, tokenizer_path


def _save_web_checkpoint(tmp_path, cfg):
    from openlocalagent.agent.pointer_head import PointerHead
    from openlocalagent.agent.tool_head import ToolHead

    model = LocalAgentLM(cfg).eval()
    checkpoint = tmp_path / "bpe.pt"
    torch.save({
        "cfg": cfg.__dict__,
        "state_dict": model.state_dict(),
        "tool_head": ToolHead(cfg.d_model).state_dict(),
        "ptr_head": PointerHead(cfg.d_model).state_dict(),
    }, checkpoint)
    return checkpoint


def test_web_export_bundles_validated_bpe_tokenizer(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_web
    from openlocalagent.model import tokenizer as tk

    tokenizer, tokenizer_path = _make_bpe(tmp_path)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, ffn_hidden=128, max_seq_len=64, name="bpe")
    checkpoint = _save_web_checkpoint(tmp_path, cfg)
    out_dir = tmp_path / "web"

    stats = export_web(
        str(checkpoint),
        str(out_dir),
        fp16=False,
        check=False,
        action_only=True,
        tokenizer_path=str(tokenizer_path),
    )

    bundled_path = out_dir / "tokenizer.json"
    meta = json.loads((out_dir / "meta.json").read_text())
    assert bundled_path.read_bytes() == tokenizer_path.read_bytes()
    assert stats["tokenizer.json"] == str(bundled_path)
    assert meta["encoding"] == "bytelevel-bpe"
    assert meta["tokenizer_file"] == "tokenizer.json"
    assert meta["model_file"] == "model.onnx"
    assert meta["action_model_file"] == "action_model.onnx"
    assert meta["model_parameters"] == cfg.estimate_params()
    assert meta["vocab_size"] == tokenizer.vocab_size
    assert meta["eos_id"] == tokenizer.eos_id == 0
    assert meta["pad_id"] == tokenizer.pad_id == 0
    assert meta["markers"]["assistant"]["ids"] == tokenizer.encode(tk.ASSISTANT)
    assert len(meta["markers"]["assistant"]["ids"]) == 1

    manifest = json.loads((out_dir / "bundle-manifest.json").read_text())
    assert manifest["config_name"] == cfg.name
    assert manifest["model_parameters"] == cfg.estimate_params()
    assert manifest["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest["checkpoint_stage"] is None
    assert manifest["checkpoint_step"] is None
    assert manifest["model_config_sha256"] == hashlib.sha256(
        json.dumps(cfg.__dict__, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    assert "bundle-manifest.json" not in manifest["artifacts"]
    for name in (
        "model.onnx",
        "action_model.onnx",
        "heads.json",
        "meta.json",
        "tokenizer.json",
    ):
        artifact = out_dir / manifest["artifacts"][name]["file"]
        assert manifest["artifacts"][name]["bytes"] == artifact.stat().st_size
        assert manifest["artifacts"][name]["sha256"] == hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest()


def test_web_export_rejects_bpe_tokenizer_hash_mismatch(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_web

    tokenizer, tokenizer_path = _make_bpe(tmp_path)
    mismatched_path = tmp_path / "mismatched-tokenizer.json"
    # Preserve a valid tokenizer while changing only its byte identity.  The exporter must check
    # the recorded hash rather than accepting any same-vocabulary file.
    mismatched_path.write_text(tokenizer_path.read_text() + "\n")
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        max_seq_len=64,
        name="bpe-mismatch",
    )
    model = LocalAgentLM(cfg).eval()
    checkpoint = tmp_path / "bpe-mismatch.pt"
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state_dict": model.state_dict(),
            "tokenizer": {
                "kind": "bpe",
                "path": str(tokenizer_path),
                "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="tokenizer artifact SHA-256"):
        export_web(
            str(checkpoint),
            str(tmp_path / "mismatch-web"),
            fp16=False,
            check=False,
            action_only=True,
            tokenizer_path=str(mismatched_path),
        )


def test_web_export_rejects_mismatched_bpe_tokenizer(tmp_path):
    from openlocalagent.inference.export.to_onnx import export_web

    tokenizer, tokenizer_path = _make_bpe(tmp_path)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size + 1, d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, ffn_hidden=128, max_seq_len=64, name="mismatch")
    checkpoint = _save_web_checkpoint(tmp_path, cfg)
    out_dir = tmp_path / "web"

    with pytest.raises(ValueError, match="vocabulary does not match"):
        export_web(
            str(checkpoint),
            str(out_dir),
            fp16=False,
            check=False,
            tokenizer_path=str(tokenizer_path),
        )
    assert not out_dir.exists()


def test_web_export_tool_head_from_json(tmp_path):
    """Prove JS can reproduce tool selection from the onnx `hidden` output + heads.json."""
    stats, m, th, ph, cfg = _make_bundle(tmp_path)
    heads = json.load(open(stats["heads.json"]))
    meta = json.load(open(stats["meta.json"]))
    sess = ort.InferenceSession(stats["model.onnx"], providers=["CPUExecutionProvider"])

    x = torch.randint(0, 256, (1, 11))
    got_logits, got_hidden = sess.run(["logits", "hidden"], {"input_ids": x.numpy()})

    # JS-side recipe: last position, matmul with tool_head weights.
    W = np.array(heads["tool_head"]["weight"], dtype=np.float32)   # (n_classes, d_model)
    b = np.array(heads["tool_head"]["bias"], dtype=np.float32)     # (n_classes,)
    last = got_hidden[:, -1]                                       # (1, d_model)
    js_logits = last @ W.T + b                                     # (1, n_classes)

    with torch.no_grad():
        ref_tool = th(torch.tensor(got_hidden[:, -1]))            # (1, 22)
    ref_tool = ref_tool.numpy()

    assert js_logits.shape == ref_tool.shape
    assert np.abs(js_logits - ref_tool).max() < 1e-3
    assert int(js_logits.argmax(-1)[0]) == int(ref_tool.argmax(-1)[0])

    # contract sanity: stop_index points at "text"; classes match meta.
    assert heads["tool_head"]["classes"][heads["tool_head"]["stop_index"]] == "text"
    assert meta["tool_classes"] == heads["tool_head"]["classes"]
    assert meta["pad_id"] == 0


def test_web_export_pointer_head_from_json(tmp_path):
    """Reproduce pointer-head span logits from heads.json (numpy) vs PyTorch."""
    stats, m, th, ph, cfg = _make_bundle(tmp_path)
    heads = json.load(open(stats["heads.json"]))
    sess = ort.InferenceSession(stats["model.onnx"], providers=["CPUExecutionProvider"])

    x = torch.randint(0, 256, (1, 9))
    _, hidden = sess.run(["logits", "hidden"], {"input_ids": x.numpy()})
    h = hidden[0]                                                  # (T, d_model)

    arg_emb = np.array(heads["pointer_head"]["arg_emb"], dtype=np.float32)
    start_W = np.array(heads["pointer_head"]["start_W"], dtype=np.float32)
    end_W = np.array(heads["pointer_head"]["end_W"], dtype=np.float32)
    arg = "query"
    i = heads["pointer_head"]["arg_idx"][arg]
    q = arg_emb[i]                                                 # (d_model,)
    qs = start_W @ q                                               # (d_model,)
    qe = end_W @ q
    s_js = h @ qs                                                  # (T,)
    e_js = h @ qe

    with torch.no_grad():
        s_ref, e_ref = ph.logits(torch.tensor(hidden),
                                 torch.tensor([i], dtype=torch.long))
    s_ref, e_ref = s_ref.numpy()[0], e_ref.numpy()[0]

    assert np.abs(s_js - s_ref).max() < 1e-3
    assert np.abs(e_js - e_ref).max() < 1e-3
