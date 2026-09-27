from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_native_webgpu_harness_is_explicit_and_fail_closed() -> None:
    html = (ROOT / "demos/webgpu/webgpu-capability.html").read_text(encoding="utf-8")
    script = (ROOT / "demos/webgpu/webgpu-capability.js").read_text(encoding="utf-8")
    assert 'window.__localAgentBenchmarkGrade = true' in html
    assert 'window.__localAgentRequestedBackend = "webgpu"' in html
    assert 'executionProviders' not in script  # provider is selected by the shared benchmark-grade app loader
    assert 'navigator.gpu.requestAdapter' in script
    assert 'environment_executed: true' in script
    assert 'notion_write' in script
    assert 'closed_loop_success: 0' in script
    assert 'driver VRAM counter unavailable' in script
    assert 'Fail-closed harness failure' in script
