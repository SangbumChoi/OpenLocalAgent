"""A `data.sources` config must run end to end, not just sample batches.

The matrix loader was wired into the training loop and tested in isolation; the run-metadata
writer still read data_cfg["shards_dir"], so the first real mixture run died on KeyError after
loading every shard. A unit test on SourceMatrix cannot see that - only a run through
pretrain.run() can, and this is that run: two tiny packed corpora, two steps, on CPU.
"""

import json
from pathlib import Path

import torch
import yaml

from openlocalagent.data.corpus.pretrain_corpus import pack_shards
from openlocalagent.model import ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer
from openlocalagent.train.pretrain import run as run_pretrain


def _tiny_corpus(root: Path, name: str, text: str) -> Path:
    shards = root / name
    pack_shards([text * 8], ByteTokenizer(), seq_len=8, shards_dir=str(shards),
                rows_per_shard=8, val_fraction=0.0)
    return shards


def test_mixture_config_runs_and_records_its_sources(tmp_path: Path):
    cfg = ModelConfig(name="mix", vocab_size=256, d_model=16, n_layers=1, n_heads=2,
                      n_kv_heads=1, ffn_hidden=32, max_seq_len=8)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(cfg.__dict__), encoding="utf-8")
    web = _tiny_corpus(tmp_path, "web", "agent tools and deterministic training ")
    math = _tiny_corpus(tmp_path, "math", "two plus two equals four minus one ")

    out_dir = tmp_path / "run"
    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text(yaml.safe_dump({
        "stage": "pretrain",
        "model_config": str(model_path),
        "data": {
            "weighting": "explicit",
            "sources": [{"name": "web", "path": str(web)},
                        {"name": "math", "path": str(math), "weight": 3.0}],
            "tokenizer": {"kind": "byte"},
        },
        "optim": {"lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0},
        "schedule": {"type": "cosine", "warmup_steps": 0, "total_steps": 2},
        "batch": {"micro_batch_size": 1, "grad_accum_steps": 4},
        "runtime": {"device": "cpu", "dtype": "fp32", "seed": 7},
        "log": {"out_dir": str(out_dir)},
    }, sort_keys=False), encoding="utf-8")

    run_pretrain(str(config_path))

    # Provenance answers "what was this trained on" for a mixture, not just for one directory.
    ck = torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["data"]["path"].startswith("mixture:")
    assert set(ck["data"]["sources"]) == {"web", "math"}
    assert ck["data"]["sources"]["math"]["share"] == 0.75

    # And the curve carries one loss per corpus, which is the point of the whole feature.
    rows = [json.loads(line) for line in (out_dir / "curve.jsonl").read_text().splitlines() if line.strip()]
    assert rows, "no curve written"
    seen = set()
    for row in rows:
        assert "sources" in row, "mixture run wrote a curve record with no per-source block"
        seen |= {k for k, v in row["sources"].items() if v.get("loss") is not None}
    assert seen <= {"web", "math"} and seen, f"unexpected/empty per-source keys: {seen}"
