import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m306-current-parent-stateful-rl-weight-transfer-v1.json")


def test_m306_weight_audit_binds_current_parent_and_recommends_body_head_split() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["base"]["sha256"] == (
        "bc1aca209ec08df1483a3c6d088366a68f8d8f4f0766e2b4350a2ef473c16361"
    )
    assert payload["target"]["sha256"] == (
        "db89b1d3d1f9dedd9d18df1a0f26e9e12bd3a02277253f06a9521e4f9e11ddc0"
    )
    compatibility = payload["compatibility"]
    assert compatibility["config_mismatches"] == {}
    assert compatibility["shape_mismatches"] == {}
    assert compatibility["tokenizer_sha256_equal"] is True
    assert compatibility["shared_tensor_count"] == 51
    assert payload["groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert payload["groups"]["embedding"]["relative_delta_l2"] > 0.01
    assert "smaller learning rate" in payload["recommendation"]["optimization"]
