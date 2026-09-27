import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m718-visual-webgpu-probe-v1.json"


def test_m718_visual_probe_is_explicit_and_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m718_visual_webgpu_probe"
    assert payload["runtime"]["webgpu"]["requested_provider"] == "webgpu"
    assert payload["runtime"]["webgpu"]["session_ready"] is True
    assert payload["runtime"]["wasm_control"]["session_ready"] is True
    assert payload["outputs"]["action"] in payload["outputs"]["action_names"]
    assert payload["verification"]["per_node_gpu_placement"] == "unknown"
    assert payload["verification"]["native_android_emulator"] is False
    assert payload["verification"]["official_androidcontrol_score"] is False
