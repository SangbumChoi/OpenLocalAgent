import pytest
import torch

from openlocalagent.model.vision import ANDROID_ACTIONS, VisualActionHead


def test_visual_action_head_emits_action_and_normalized_pointer() -> None:
    head = VisualActionHead(32)
    action, pointer = head(torch.randn(3, 32), torch.randn(3, 9, 32))
    assert action.shape == (3, len(ANDROID_ACTIONS))
    assert pointer.shape == (3, 2)
    assert torch.all((pointer >= 0) & (pointer <= 1))


def test_visual_action_head_rejects_mismatched_width() -> None:
    head = VisualActionHead(32)
    with pytest.raises(ValueError, match="width"):
        head(torch.randn(3, 31), torch.randn(3, 9, 32))
