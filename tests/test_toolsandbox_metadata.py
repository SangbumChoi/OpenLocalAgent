from pathlib import Path

from scripts.profile_toolsandbox_metadata import profile


def _fixture(root: Path) -> None:
    scenario_dir = root / "tool_sandbox/scenarios"
    common_dir = root / "tool_sandbox/common"
    scenario_dir.mkdir(parents=True)
    common_dir.mkdir(parents=True)
    scenario_template = """
from tool_sandbox.common.execution_context import ScenarioCategories
from tool_sandbox.common.scenario import Scenario

def named_fixture():
    return {
        "one": Scenario(name="{prefix}_one", tool_allow_list=["alpha"], categories=[ScenarioCategories.CANONICALIZATION]),
        "two": Scenario(name="{prefix}_two", tool_allow_list=["beta", "gamma"], categories=[ScenarioCategories.STATE_DEPENDENCY]),
    }
"""
    for filename in (
        "single_tool_call_scenarios.py",
        "multiple_tool_call_scenarios.py",
        "multiple_user_turn_scenarios.py",
        "insufficient_information_scenarios.py",
    ):
        (scenario_dir / filename).write_text(
            scenario_template.replace("{prefix}", filename.removesuffix("_scenarios.py")),
            encoding="utf-8",
        )
    (common_dir / "execution_context.py").write_text("class ScenarioCategories: pass\n", encoding="utf-8")
    (scenario_dir / "__init__.py").write_text("# runtime augmentation\n", encoding="utf-8")
    (root / "README.md").write_text("public metadata\n", encoding="utf-8")
    (root / "LICENSE").write_text("license\n", encoding="utf-8")


def test_toolsandbox_profile_is_ast_only_and_hash_bound(tmp_path: Path) -> None:
    _fixture(tmp_path)
    receipt = profile(tmp_path, revision="fixture")
    assert receipt["coverage"]["base_scenarios"] == 8
    assert receipt["coverage"]["source_level_augmented_scenarios"] == 64
    assert receipt["coverage"]["category_tokens"]["CANONICALIZATION"] == 4
    assert receipt["coverage"]["unique_tools"] == 3
    assert receipt["source"]["tools_executed"] is False
    assert len(receipt["receipt_self_sha256"]) == 64
