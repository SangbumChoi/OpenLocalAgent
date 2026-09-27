import torch

from scripts.train_stateful_productivity_rl import _preserve_deployment_heads


def test_rl_child_preserves_pointer_argument_metadata(tmp_path) -> None:
    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    ptr_args = ["message", "subject", "body"]
    torch.save(
        {
            "ptr_args": ptr_args,
            "ptr_head": {"arg_emb.weight": torch.zeros(len(ptr_args), 2)},
        },
        parent,
    )
    torch.save(
        {"ptr_head": {"arg_emb.weight": torch.ones(len(ptr_args), 2)}},
        child,
    )

    _preserve_deployment_heads(parent, child)
    payload = torch.load(child, map_location="cpu", weights_only=False)
    assert payload["ptr_args"] == ptr_args
    assert payload["ptr_head"]["arg_emb.weight"].shape == (len(ptr_args), 2)
