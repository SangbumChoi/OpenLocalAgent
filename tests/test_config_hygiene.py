"""Configs must be complete, self-chained, and not accumulate orphans.

configs/model held 31 files for a seven-tier ladder. Most of the surplus was the provenance of a
published receipt and had to stay; two were referenced by nothing at all. Telling those apart
required grepping the campaign archive, which is exactly the work a test should do once.

The chaining assertion is the expensive one to get wrong: an arm that inherits its base's out_dir
trains over the base's checkpoint and produces a receipt attributed to the wrong run.
"""

from pathlib import Path

import pytest
import yaml

CONFIGS = Path("configs")
STAGES = ("pretrain", "midtrain", "posttrain")


def stage_configs(stage: str) -> list[Path]:
    return sorted(CONFIGS.joinpath(stage).glob("*.yaml"))


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def tier_of(path: Path) -> str | None:
    """`la-93m-s2.yaml` -> `93m-s2`; None for the non-ladder configs."""
    return path.stem[3:] if path.stem.startswith("la-") else None


LADDER = [p for stage in STAGES for p in stage_configs(stage) if tier_of(p)]


def test_the_ladder_is_not_empty():
    assert LADDER, "no la-* configs; the assertions below would be vacuous"


@pytest.mark.parametrize("path", LADDER, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_stage_writes_to_its_own_run(path: Path):
    """An arm inheriting its base's out_dir overwrites the base's checkpoint, silently."""
    tier = tier_of(path)
    config = load(path)
    out_dir = config["log"]["out_dir"]
    assert out_dir == f"results/runs/{path.parent.name}-la-{tier}", (
        f"{path} writes to {out_dir}, not its own run directory"
    )


@pytest.mark.parametrize("path", [p for p in LADDER if p.parent.name != "pretrain"],
                         ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_later_stage_chains_from_its_own_tier(path: Path):
    """midtrain must resume this tier's pretrain, not the tier it was copied from."""
    tier = tier_of(path)
    config = load(path)
    parent = "pretrain" if path.parent.name == "midtrain" else "midtrain"
    # An ablation may chain from its control's run instead of its own tier - the point of
    # `la-93m-matrix-nomid` is that no midtrain of its own exists - but only when it says so.
    ablation = config.get("ablation") or {}
    if ablation.get("init_from_control"):
        tier = ablation["of"].removeprefix("la-")
        assert tier_of(Path(f"la-{tier}.yaml")), f"ablation.of must name a ladder tier: {tier}"
        if ablation.get("skips") == "midtrain":
            assert path.parent.name == "posttrain"
            parent = "pretrain"
    assert config["init_from"] == f"results/runs/{parent}-la-{tier}/latest.pt"


@pytest.mark.parametrize("path", stage_configs("pretrain"),
                         ids=lambda p: p.stem)
def test_pretrain_names_exactly_one_corpus_source(path: Path):
    """Either one packed corpus or a mixture - both would be ambiguous about what was trained on."""
    data = load(path)["data"]
    has_dir, has_mixture = "shards_dir" in data, bool(data.get("sources"))
    assert has_dir != has_mixture, (
        f"{path} declares {'both' if has_dir else 'neither'} shards_dir and sources"
    )


@pytest.mark.parametrize("path", [p for p in stage_configs("pretrain") if tier_of(p)],
                         ids=lambda p: p.stem)
def test_every_tier_has_all_three_stages(path: Path):
    tier = tier_of(path)
    for stage in ("midtrain", "posttrain"):
        assert (CONFIGS / stage / f"la-{tier}.yaml").is_file(), (
            f"la-{tier} has a {path.parent.name} config but no {stage}: a spine that cannot finish"
        )


def test_model_configs_are_referenced_or_are_provenance():
    """A model config nobody references and no receipt cites is dead weight.

    The campaign archive counts: la64k-8k and la64k-16k are referenced only by published receipts,
    and deleting them would leave a published number with no config to explain it.
    """
    import subprocess

    orphans = []
    for path in sorted((CONFIGS / "model").glob("*.yaml")):
        hits = subprocess.run(
            ["grep", "-rl", path.stem, "src", "tests", "scripts", "demos", "configs",
             "docs", "README.md", "AGENTS.md"],
            capture_output=True, text=True).stdout.splitlines()
        if not [h for h in hits if h != str(path)]:
            orphans.append(path.name)
    assert not orphans, (
        f"model configs referenced by nothing and citing no receipt: {orphans}. "
        "Delete them, or point something at them."
    )


@pytest.mark.parametrize("path", [p for stage in ("midtrain", "posttrain") for p in stage_configs(stage)],
                         ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_context_fitted_paths_carry_exactly_one_tokenizer_tag(path: Path):
    """`fit2048-<tag>-<tag>-toucan.jsonl` killed la-45m-matrix's midtrain on a missing file.

    The arm derivation applied an explicit rename and then a generic fit2048-* regex on top of
    it. A doubled tag is never a real file, so any occurrence is a generation bug, not data.
    """
    import re

    body = path.read_text(encoding="utf-8")
    doubled = re.findall(r"fit\d+-([0-9a-f]{8})-\1-", body)
    assert not doubled, f"{path} references a context-fitted file with a doubled tokenizer tag"


@pytest.mark.parametrize("path", LADDER, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_ladder_stage_resumes_its_own_checkpoint(path: Path):
    """The queue relies on runtime.resume to make re-running a finished stage a no-op. Posttrain
    configs lacked it, so every queue relaunch retrained posttrain from scratch - nine hours of
    GPU 1 in one night, and receipts silently rewritten by a second training run."""
    assert (load(path).get("runtime") or {}).get("resume") is True, path
