"""Every script in scripts/ says what it does in its first word.

scripts/ was the directory that accreted worst before the refactor - 396 files across twenty ad-hoc
prefixes, where finding "the thing that builds the corpus" meant reading names. A closed verb set
makes one `ls scripts/<verb>_*` answer that, and this test is what keeps the set closed: a new
script with an unlisted verb fails here rather than starting the next twenty.
"""

from pathlib import Path

import pytest

SCRIPTS = Path("scripts")

#: The verbs, in pipeline order. Each names a distinct kind of work, so the right one is never
#: ambiguous: data comes in (fetch), becomes rows (normalize), becomes a dataset (build), trains
#: (train), is scored (eval), is written up (analyze), or leaves (export/archive).
VERBS = {
    "fetch": "acquire bytes from an external source",
    "normalize": "turn one external source into Conversation rows",
    "build": "derive a dataset or artifact from data already local",
    "train": "run or launch training",
    "eval": "score a model",
    "analyze": "produce a figure or report from results",
    "export": "emit an artifact for somewhere else",
    "archive": "retire a run and its receipt",
}

#: One-command entry points that are named for what they are, not what they do.
ENTRY_POINTS = {"speedrun.sh"}


def script_files() -> list[Path]:
    return sorted(p for p in SCRIPTS.iterdir()
                  if p.is_file() and p.suffix in {".py", ".sh"})


def test_scripts_directory_is_not_empty():
    assert len(script_files()) > 10, "the audit below is only meaningful against real files"


@pytest.mark.parametrize("path", script_files(), ids=lambda p: p.name)
def test_every_script_starts_with_a_known_verb(path: Path):
    if path.name in ENTRY_POINTS:
        return
    verb = path.stem.split("_")[0]
    assert verb in VERBS, (
        f"{path.name} starts with {verb!r}, which is not one of {sorted(VERBS)}. "
        "Rename it to the verb that describes what it does, or make the case for a new verb "
        "in docs/CONVENTIONS.md - do not add a twenty-first prefix."
    )


@pytest.mark.parametrize("path", script_files(), ids=lambda p: p.name)
def test_every_script_says_what_it_is_for(path: Path):
    """A first line of prose, so `head -3` is enough to know whether this is the script you want."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()[:6]]
    prose = [line for line in lines if line.startswith(('"""', "#")) and len(line) > 8]
    assert prose, f"{path.name} has no docstring or header comment in its first six lines"


def test_no_verb_is_unused():
    """A verb nobody uses is a verb nobody will pick correctly."""
    used = {p.stem.split("_")[0] for p in script_files() if p.name not in ENTRY_POINTS}
    unused = set(VERBS) - used
    assert not unused, f"verbs declared but unused: {sorted(unused)}"


# --- shell scripts are held to the same standard as the Python ---

SHELL = [p for p in script_files() if p.suffix == ".sh"]


@pytest.mark.parametrize("path", SHELL, ids=lambda p: p.name)
def test_shell_declares_bash(path: Path):
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first == "#!/usr/bin/env bash", (
        f"{path.name} starts with {first!r}; use '#!/usr/bin/env bash' so it does not depend on "
        "whichever shell happens to invoke it"
    )


@pytest.mark.parametrize("path", SHELL, ids=lambda p: p.name)
def test_shell_fails_fast(path: Path):
    """`set -u` alone lets a failed download scroll past and the script exit 0.

    That is not hypothetical: fetch_midtrain_open.sh ran with `set -u`, several curls 404'd, and
    the run reported success with files missing.
    """
    body = path.read_text(encoding="utf-8")
    assert "set -euo pipefail" in body, f"{path.name} does not `set -euo pipefail`"


@pytest.mark.parametrize("path", SHELL, ids=lambda p: p.name)
def test_shell_has_no_absolute_machine_paths(path: Path):
    """A script pinned to /home/jovyan or /Users only runs on one machine."""
    body = path.read_text(encoding="utf-8")
    for needle in ("/home/jovyan", "/Users/"):
        assert needle not in body, (
            f"{path.name} hardcodes {needle}; resolve the repository root from BASH_SOURCE and "
            "take data roots from an environment variable with a sensible default"
        )
