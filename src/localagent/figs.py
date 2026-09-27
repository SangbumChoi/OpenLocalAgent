"""Save experiment plots into the repo's central ``figures/`` gallery.

    from localagent.figs import savefig
    savefig(fig, "my_experiment")        # -> figures/my_experiment.png

Every experiment should use this so all result plots land in one committed folder.
"""

from __future__ import annotations

import os


def figures_dir() -> str:
    """Repo ``figures/`` — overridable via $LOCALAGENT_FIGURES; else found from pyproject.toml."""
    if (d := os.environ.get("LOCALAGENT_FIGURES")):
        return d
    p = os.path.abspath(os.getcwd())
    while p != os.path.dirname(p):
        if os.path.exists(os.path.join(p, "pyproject.toml")):
            return os.path.join(p, "figures")
        p = os.path.dirname(p)
    return "figures"


def savefig(fig, name: str, dpi: int = 120) -> str:
    d = figures_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name if name.endswith(".png") else name + ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
