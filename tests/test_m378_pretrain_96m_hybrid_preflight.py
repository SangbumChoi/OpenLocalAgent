from __future__ import annotations

import json
from pathlib import Path

import yaml

from localagent.train.stage_data import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m378-pretrain-96m-hybrid-one-microbatch-v1.json"


def test_m378_receipt_proves_one_isolated_96m_update() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("receipt_self_sha256")
    assert expected == canonical_sha256(body)
    assert payload["kind"] == "localagent_one_update_training_preflight"
    assert payload["status"] == "passed"
    assert payload["model"] == {
        "exact_parameters": 95_320_448,
        "max_seq_len": 4096,
        "name": "webgpu-96m-hybrid",
        "vocab_size": 16_384,
    }
    contract = payload["effective"]["contract"]
    assert contract["optimizer_updates"] == 1
    assert contract["production_schedule_total_steps"] == 2909
    assert contract["micro_batch_size"] == 1
    assert contract["grad_accum_steps"] == 1
    assert contract["evaluation"] == "disabled"
    assert payload["metrics"]["execution"]["resolved_device"] == "cpu"
    assert payload["metrics"]["execution"]["resolved_dtype"] == "fp32"
    assert payload["metrics"]["steps_completed"] == 1
    assert payload["metrics"]["token_accounting"]["input_tokens"] == 2048
    assert payload["source"]["source_artifacts_untouched"] is True
    assert payload["source"]["production_checkpoint_untouched"] is True
    assert payload["error"] is None


def test_m378_derivative_is_explicitly_one_microbatch() -> None:
    production = yaml.safe_load(
        (ROOT / "configs/train/pretrain-paper-tier-96m-hybrid.yaml").read_text(
            encoding="utf-8"
        )
    )
    preflight = yaml.safe_load(
        (ROOT / "configs/train/pretrain-paper-tier-96m-hybrid-preflight.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert production["batch"]["grad_accum_steps"] == 16
    assert preflight["batch"]["grad_accum_steps"] == 1
    assert preflight["model_config"] == production["model_config"]
    assert preflight["data"] == production["data"]
    assert preflight["optim"] == production["optim"]
