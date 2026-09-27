import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m724-current-checkpoint-visual-webgpu-v1.json"


def test_m724_current_graph_runs_in_explicit_webgpu_probe() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_m724_current_checkpoint_visual_webgpu_probe"
    assert payload["checkpoint_sha256"].startswith("6a6520")
    assert payload["runtime"]["requested_provider"] == "webgpu"
    assert payload["runtime"]["session_ready"] is True
    assert payload["graph"]["cpu_onnx_parity"]["passed"] is True
    assert payload["verification"]["native_android_emulator"] is False
