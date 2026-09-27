"""Versioned schema overlay and train-only slots for the paper synthetic-data v2 profile.

The frozen v1 generator embeds ``STANDARD_TOOLS`` in every row, so changing that global registry
would change all legacy JSONL bytes.  Paper v2 instead clones the registry and overlays two
optional properties on the existing ``scroll`` tool.  Tool names and ordering stay identical for
the structured heads, while v2 conversations can exercise missing JSON boolean/number types.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

from localagent.data.schema import ToolSpec

PAPER_TRAIN_V2_MODE = "paper_train_v2"
PAPER_TRAIN_V2_MODE_VERSION = 2

# Non-integral values cannot collide with the frozen integer eval slots.  Quarter increments keep
# JSON rendering short and exactly reproducible while spanning small and large scroll magnitudes.
PAPER_V2_SCROLL_AMOUNTS_TRAIN = tuple(
    round(0.25 + 0.5 * index, 2) for index in range(120)
)

# The boolean primitive domain itself cannot be train/eval-disjoint.  These *language cues* are
# train-only slots; the manifest states honestly that boolean values are schema primitives.
PAPER_V2_SCROLL_BOOLEAN_CUES_TRAIN = (
    ("enable smooth scrolling", True),
    ("turn on smooth scrolling", True),
    ("activate smooth scrolling", True),
    ("disable smooth scrolling", False),
    ("turn off smooth scrolling", False),
    ("leave smooth scrolling off", False),
)

PAPER_TRAIN_V2_SLOT_POOLS: dict[str, tuple[object, ...]] = {
    "paper_v2_scroll_amounts_train": PAPER_V2_SCROLL_AMOUNTS_TRAIN,
    "paper_v2_scroll_boolean_cues_train": tuple(
        cue for cue, _value in PAPER_V2_SCROLL_BOOLEAN_CUES_TRAIN
    ),
}


def build_paper_train_v2_tools(
    standard_tools: Sequence[ToolSpec],
) -> list[ToolSpec]:
    """Clone the standard registry and add optional number/boolean fields to ``scroll``.

    Existing v1 scroll calls remain schema-valid because only ``direction`` stays required.
    """

    tools = copy.deepcopy(list(standard_tools))
    scroll_specs = [tool for tool in tools if tool.name == "scroll"]
    if len(scroll_specs) != 1:
        raise ValueError("paper_train_v2 requires exactly one standard scroll tool")
    scroll = scroll_specs[0]
    parameters = scroll.parameters
    properties = parameters.get("properties")
    if not isinstance(properties, dict) or "direction" not in properties:
        raise ValueError("paper_train_v2 scroll schema lacks the direction property")
    if "amount" in properties or "smooth" in properties:
        raise ValueError("paper_train_v2 scroll schema overlay conflicts with existing fields")
    properties["amount"] = {
        "type": "number",
        "description": "Distance to scroll in screen lengths.",
    }
    properties["smooth"] = {
        "type": "boolean",
        "description": "Whether to animate the scroll smoothly.",
    }
    return tools


__all__ = [
    "PAPER_TRAIN_V2_MODE",
    "PAPER_TRAIN_V2_MODE_VERSION",
    "PAPER_TRAIN_V2_SLOT_POOLS",
    "PAPER_V2_SCROLL_AMOUNTS_TRAIN",
    "PAPER_V2_SCROLL_BOOLEAN_CUES_TRAIN",
    "build_paper_train_v2_tools",
]
