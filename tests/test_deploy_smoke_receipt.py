import json
from pathlib import Path


RECEIPT = Path(__file__).parents[1] / "docs/paper/results/raw/m24-local-deployment-smoke-v1.json"


def test_local_deployment_smoke_receipt_is_explicitly_non_native() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["kind"] == "localagent_local_deployment_smoke"
    assert payload["environment"]["environment_executed"] is False
    assert payload["environment"]["external_accounts"] is False
    assert payload["summary"]["cases"] == 10
    assert payload["summary"]["exact_tool"] == 4
    assert "not a browser" in payload["claim_boundary"]
