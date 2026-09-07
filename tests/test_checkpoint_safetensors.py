"""A training payload must survive safetensors intact - tensors, scalars, tuples and int keys.

safetensors stores a flat tensor dict plus a str->str header and nothing else, while a checkpoint
is a tree: cfg and lineage beside state_dict, the optimizer's two moments per parameter, and four
kinds of RNG state. The round trip is the whole feature, so these assert the parts that fail
silently rather than loudly - a tuple degraded to a list, or an optimizer keyed by "0" instead of
0, both load without error and both discard the state they were meant to carry.
"""

from pathlib import Path

import pytest
import torch

from openlocalagent.train.checkpoint import load_safetensors, save_safetensors

pytest.importorskip("safetensors")


def full_payload() -> dict:
    """The shape measured on results/runs/pretrain-la-93m/latest.pt, in miniature."""
    return {
        "cfg": {"name": "la-93m", "vocab_size": 16384, "d_model": 64},
        "state_dict": {"embed.weight": torch.randn(8, 4), "out.bias": torch.zeros(4)},
        "optimizer": {
            "state": {0: {"step": torch.tensor(1600.0),
                          "exp_avg": torch.randn(8, 4),
                          "exp_avg_sq": torch.rand(8, 4)}},
            "param_groups": [{"lr": 3e-4, "weight_decay": 0.1, "params": [0]}],
        },
        "step": 15999,
        "tokens_seen": 2097150787,
        "loss_history": [3.2, 2.9, 2.4],
        "rng_state": ("MT19937", [1, 2, 3], None),
        "torch_rng_state": torch.zeros(16, dtype=torch.uint8),
        "cuda_rng_state_all": [torch.ones(8, dtype=torch.uint8)],
        "mps_rng_state": None,
        "grad_scaler": None,
        "stage": "pretrain",
        "lineage": {"tokenizer_sha256": "236945b5" + "0" * 56},
        "tokenizer": {"kind": "bpe", "sha256": "236945b5" + "0" * 56},
    }


def test_round_trip_preserves_every_tensor(tmp_path: Path):
    original = full_payload()
    loaded = load_safetensors(save_safetensors(original, tmp_path / "ck.safetensors"))
    for key in ("embed.weight", "out.bias"):
        assert torch.equal(loaded["state_dict"][key], original["state_dict"][key])
    for moment in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(loaded["optimizer"]["state"][0][moment],
                           original["optimizer"]["state"][0][moment])
    assert torch.equal(loaded["torch_rng_state"], original["torch_rng_state"])
    assert torch.equal(loaded["cuda_rng_state_all"][0], original["cuda_rng_state_all"][0])


def test_round_trip_preserves_scalars_and_none(tmp_path: Path):
    original = full_payload()
    loaded = load_safetensors(save_safetensors(original, tmp_path / "ck.safetensors"))
    assert loaded["step"] == 15999 and loaded["tokens_seen"] == 2097150787
    assert loaded["stage"] == "pretrain" and loaded["cfg"] == original["cfg"]
    assert loaded["loss_history"] == [3.2, 2.9, 2.4]
    assert loaded["mps_rng_state"] is None and loaded["grad_scaler"] is None
    assert loaded["lineage"]["tokenizer_sha256"].startswith("236945b5")


def test_tuple_stays_a_tuple(tmp_path: Path):
    """python's RNG state is a tuple; JSON would make it a list and random.setstate would refuse."""
    loaded = load_safetensors(save_safetensors(full_payload(), tmp_path / "ck.safetensors"))
    assert isinstance(loaded["rng_state"], tuple)
    assert loaded["rng_state"][0] == "MT19937" and loaded["rng_state"][2] is None


def test_optimizer_state_keys_stay_integers(tmp_path: Path):
    """Keyed by "0" the optimizer matches no parameter and resumes with the moments discarded."""
    loaded = load_safetensors(save_safetensors(full_payload(), tmp_path / "ck.safetensors"))
    assert set(loaded["optimizer"]["state"]) == {0}
    assert loaded["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-4)


def test_optimizer_actually_reloads_into_torch(tmp_path: Path):
    """The end that matters: torch must accept the state, not merely round-trip the bytes."""
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()

    payload = {"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "step": 1}
    loaded = load_safetensors(save_safetensors(payload, tmp_path / "ck.safetensors"))

    fresh_model = torch.nn.Linear(4, 3)
    fresh = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_model.load_state_dict(loaded["state_dict"])
    fresh.load_state_dict(loaded["optimizer"])
    assert fresh.state_dict()["state"][0]["exp_avg"].shape == (3, 4)


def test_write_is_atomic(tmp_path: Path):
    """Interrupted saves must not leave a half file that the next resume would load."""
    target = tmp_path / "ck.safetensors"
    save_safetensors(full_payload(), target)
    assert target.exists()
    assert not list(tmp_path.glob("*.tmp")), "temp file survived the save"


def test_shared_storage_does_not_break_the_write(tmp_path: Path):
    """Tied embeddings alias one storage; safetensors rejects that unless it is cloned."""
    shared = torch.randn(4, 4)
    payload = {"state_dict": {"embed.weight": shared, "lm_head.weight": shared}, "step": 0}
    loaded = load_safetensors(save_safetensors(payload, tmp_path / "ck.safetensors"))
    assert torch.equal(loaded["state_dict"]["embed.weight"], shared)
    assert torch.equal(loaded["state_dict"]["lm_head.weight"], shared)
