import json
from pathlib import Path


def test_m116_receipt_is_pinned_and_keeps_proxy_boundary() -> None:
    receipt = json.loads(
        Path(
            "docs/paper/results/raw/m116-mcpmark-routing-transfer-v1.json"
        ).read_text(encoding="utf-8")
    )
    source = receipt["source"]
    assert source["revision"] == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert source["standard_rows"] == 169
    assert source["easy_rows"] == 70
    assert source["training_used"] is False
    assert source["mcp_servers_executed"] is False
    assert source["verifiers_executed"] is False
    assert receipt["reports"]["warm_standard"]["route_accuracy"] == 0.14792899408284024
    assert receipt["reports"]["random_standard"]["route_accuracy"] == 0.14792899408284024
    assert receipt["reports"]["warm_standard"]["service_accuracy"]["playwright"] == 1.0
    assert receipt["reports"]["warm_standard"]["service_accuracy"]["notion"] == 0.0
    assert "not live MCP execution" in receipt["claim_boundary"]
