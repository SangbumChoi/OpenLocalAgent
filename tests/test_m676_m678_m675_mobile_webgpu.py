import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load(name: str) -> dict:
    path = ROOT / "docs/paper/results/raw" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual
    return payload


def test_m676_mobilegym_binds_the_androidcontrol_child() -> None:
    payload = _load("m676-m675-mobilegym-native-v1.json")
    assert payload["kind"] == "localagent_m676_m675_mobilegym_native_receipt"
    assert payload["checkpoint"]["sha256"].startswith("91eb9696")
    assert payload["official_split_verified"] is True
    assert payload["task_count"] == 256
    assert payload["result"]["passed_tasks"] == 1


def test_m677_m678_bind_the_androidcontrol_child_webgpu_release() -> None:
    preparation = _load("m677-m675-hf-space-preparation-v1.json")
    adoption = _load("m678-m675-webgpu-adoption-v1.json")
    assert preparation["kind"] == "localagent_m677_m675_hf_space_preparation"
    assert preparation["published"] is False
    assert preparation["checkpoint"]["sha256"] == adoption["checkpoint"]["sha256"]
    assert adoption["kind"] == "localagent_m678_m675_webgpu_adoption"
    assert adoption["native_webgpu"]["evaluated_cases"] == 3
    assert adoption["native_webgpu"]["exact_actions"] == 3
    assert adoption["native_webgpu"]["tokens_per_second_p50"] > 900
    assert adoption["native_webgpu"]["external_side_effects_executed"] is False
    assert adoption["adoption"]["public_model_published"] is False
