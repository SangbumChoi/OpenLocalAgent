"""Focused contracts for shared language-model batch collation."""

from __future__ import annotations

import pytest
import torch

from openlocalagent.data.render import IGNORE
from openlocalagent.train.loop import pad_batch


def test_pad_batch_can_right_pad_to_an_exact_post_shift_input_width() -> None:
    rows = [
        ([10, 11, 12], [IGNORE, 21, 22]),
        ([13, 14], [IGNORE, 31]),
    ]

    x, y = pad_batch(rows, pad_id=0, device="cpu", pad_to_input_tokens=4)

    assert x.shape == y.shape == (2, 4)
    assert torch.equal(
        x,
        torch.tensor(
            [
                [10, 11, 12, 0],
                [13, 14, 0, 0],
            ]
        ),
    )
    assert torch.equal(
        y,
        torch.tensor(
            [
                [21, 22, IGNORE, IGNORE],
                [31, IGNORE, IGNORE, IGNORE],
            ]
        ),
    )


def test_pad_batch_fixed_width_never_truncates_oversized_rows() -> None:
    rows = [([10, 11, 12, 13], [IGNORE, 21, 22, 23])]

    with pytest.raises(
        ValueError,
        match=r"required=3, configured=2",
    ):
        pad_batch(rows, pad_id=0, device="cpu", pad_to_input_tokens=2)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_pad_batch_fixed_width_must_be_a_positive_integer(value) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        pad_batch(
            [([10, 11], [IGNORE, 21])],
            pad_id=0,
            device="cpu",
            pad_to_input_tokens=value,
        )


def test_pad_batch_default_behavior_is_unchanged() -> None:
    rows = [
        ([10, 11, 12], [IGNORE, 21, 22]),
        ([13, 14], [IGNORE, 31]),
    ]

    legacy_x, legacy_y = pad_batch(rows, pad_id=0, device="cpu")
    explicit_x, explicit_y = pad_batch(
        rows,
        pad_id=0,
        device="cpu",
        pad_to_input_tokens=None,
    )

    assert legacy_x.shape == legacy_y.shape == (2, 2)
    assert torch.equal(legacy_x, explicit_x)
    assert torch.equal(legacy_y, explicit_y)
