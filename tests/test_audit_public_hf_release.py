import hashlib
import json
from pathlib import Path

from scripts.audit_public_hf_release import build_manifest


def test_public_release_audit_marks_legacy_model_as_not_current(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"current")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_raw = b"published-model"
    demo_raw = b"published-demo"
    model_revision = "a" * 40
    space_revision = "b" * 40
    model_metadata = {
        "private": False,
        "sha": model_revision,
        "siblings": [
            {"rfilename": "model.safetensors"},
            {"rfilename": "config.json"},
        ],
    }
    space_metadata = {
        "private": False,
        "sha": space_revision,
        "siblings": [{"rfilename": "model.fp16.onnx"}],
    }
    config = {"parameter_count": 123, "checkpoint_sha256": "0" * 64}
    responses = {
        "https://huggingface.co/api/models/example/model": json.dumps(model_metadata).encode(),
        "https://huggingface.co/api/spaces/example/demo": json.dumps(space_metadata).encode(),
        f"https://huggingface.co/example/model/resolve/{model_revision}/model.safetensors?download=true": model_raw,
        f"https://huggingface.co/spaces/example/demo/resolve/{space_revision}/model.fp16.onnx?download=true": demo_raw,
        f"https://huggingface.co/example/model/resolve/{model_revision}/config.json?download=true": json.dumps(config).encode(),
    }
    manifest = build_manifest(
        model_repo="example/model",
        space_repo="example/demo",
        checkpoint=checkpoint,
        fetch=responses.__getitem__,
    )
    assert manifest["verification"]["local_checkpoint_sha256"] == checkpoint_sha
    assert manifest["verification"]["current_checkpoint_match"] is False
    assert manifest["current_checkpoint_sha256"] == "0" * 64
    assert manifest["model"]["sha256"] == hashlib.sha256(model_raw).hexdigest()
    assert manifest["demo"]["model_graph"]["bytes"] == len(demo_raw)
    assert len(manifest["receipt_self_sha256"]) == 64
