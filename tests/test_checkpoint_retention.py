"""Retained checkpoints must be free to write, bounded on disk, and reachable after a bad save.

The behaviour under test exists because a run directory used to hold exactly one point in time.
When two queue copies interleaved writes into posttrain-la-300m, "the step before the corruption"
did not exist and the entire run was deleted. These assertions are that scenario, in miniature.
"""

from pathlib import Path

import pytest

from openlocalagent.train.checkpoint import CheckpointStore, Retention, snapshot_name


def write_latest(store: CheckpointStore, payload: bytes) -> None:
    """Mimic a stage's atomic save: temp file, then replace - which makes a NEW inode."""
    store.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = store.latest.with_suffix(".pt.tmp")
    tmp.write_bytes(payload)
    tmp.replace(store.latest)


def test_disabled_by_default_so_disk_use_cannot_change_underneath_a_config():
    assert not Retention().enabled
    assert Retention.from_log_config({}).every == 0


def test_reads_policy_from_log_config():
    policy = Retention.from_log_config({"snapshot_every": 2000, "snapshot_keep": 5})
    assert policy.enabled and policy.every == 2000 and policy.keep == 5


@pytest.mark.parametrize("step,expected", [(0, False), (1999, False), (2000, True), (4000, True)])
def test_due_lands_on_boundaries_only(step, expected):
    assert Retention(every=2000).due(step) is expected


def test_snapshot_is_a_hard_link_so_it_costs_no_bytes(tmp_path: Path):
    """The whole design rests on this: a snapshot must not copy a 7.9 GB checkpoint."""
    store = CheckpointStore(tmp_path, Retention(every=10))
    write_latest(store, b"step-10 weights and optimizer")
    snap = store.retain(10)
    assert snap is not None and snap.name == snapshot_name(10)
    assert snap.stat().st_ino == store.latest.stat().st_ino, "snapshot copied instead of linked"


def test_snapshot_survives_the_next_save(tmp_path: Path):
    """latest.pt is replaced, not written in place, so the old inode stays alive under its name."""
    store = CheckpointStore(tmp_path, Retention(every=10))
    write_latest(store, b"step-10")
    store.retain(10)
    write_latest(store, b"step-20")
    assert (tmp_path / snapshot_name(10)).read_bytes() == b"step-10"
    assert store.latest.read_bytes() == b"step-20"


def test_retention_is_bounded(tmp_path: Path):
    """A 700M checkpoint is 7.9 GB; an unbounded policy fills a shared volume."""
    store = CheckpointStore(tmp_path, Retention(every=10, keep=3))
    for step in range(10, 110, 10):
        write_latest(store, f"step-{step}".encode())
        store.retain(step)
    kept = [p.name for p in store.snapshots()]
    assert kept == [snapshot_name(s) for s in (80, 90, 100)]


def test_nothing_retained_when_policy_is_off(tmp_path: Path):
    store = CheckpointStore(tmp_path, Retention())
    write_latest(store, b"x")
    assert store.retain(100) is None and store.snapshots() == []


def test_resume_falls_back_to_a_known_good_step(tmp_path: Path):
    """The recovery path: latest.pt is corrupt or unattributable, so restart from before it."""
    store = CheckpointStore(tmp_path, Retention(every=10, keep=5))
    for step in (10, 20, 30):
        write_latest(store, f"step-{step}".encode())
        store.retain(step)
    write_latest(store, b"corrupt - two writers")
    assert store.resume_from() == store.latest
    assert store.resume_from(step=25).read_bytes() == b"step-20"
    assert store.resume_from(step=5) is None


def test_retain_is_idempotent(tmp_path: Path):
    """A resumed run re-reaching the same step must not fail or double-count."""
    store = CheckpointStore(tmp_path, Retention(every=10))
    write_latest(store, b"a")
    first = store.retain(10)
    assert store.retain(10) == first
    assert len(store.snapshots()) == 1


def test_retain_without_a_checkpoint_is_a_noop(tmp_path: Path):
    assert CheckpointStore(tmp_path, Retention(every=10)).retain(10) is None


def test_policy_states_its_own_disk_cost(tmp_path: Path):
    """So a config can be costed before it runs, not when the volume fills."""
    seven_point_nine_gb = 7_900_000_000
    assert Retention(every=2000, keep=3).bytes_for(seven_point_nine_gb) == 23_700_000_000
    assert Retention().bytes_for(seven_point_nine_gb) == 0


def test_unstamped_files_are_never_treated_as_snapshots(tmp_path: Path):
    store = CheckpointStore(tmp_path, Retention(every=10, keep=1))
    write_latest(store, b"a")
    (tmp_path / "best.pt").write_bytes(b"someone else's file")
    store.retain(10)
    assert store.prune() == []
    assert (tmp_path / "best.pt").exists(), "pruning touched a file it does not own"


def test_weights_only_snapshot_drops_the_optimizer_and_says_so(tmp_path: Path):
    """Measured on la-700m: state_dict 2.80 GB, AdamW state 5.61 GB - the optimizer is 67%.

    Two fp32 moments per parameter is twice the weights, which is why a training checkpoint is
    ~6x a released inference artifact of the same size (Qwen3-0.6B ships 1.5 GB of bf16 weights
    and no optimizer). Dropping it makes a snapshot re-scoreable but not resumable, and the
    payload has to admit that rather than look like a full checkpoint.
    """
    import torch

    store = CheckpointStore(tmp_path, Retention(every=10, weights_only=True))
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"w": torch.zeros(64)},
                "optimizer": {"state": {0: {"exp_avg": torch.zeros(64),
                                            "exp_avg_sq": torch.zeros(64)}}},
                "step": 10, "tokenizer": {"sha256": "abc"}}, store.latest)

    snap = store.retain(10)
    assert snap is not None
    payload = torch.load(snap, map_location="cpu", weights_only=False)
    assert "optimizer" not in payload
    assert payload["snapshot_kind"] == "weights_only" and payload["resumable"] is False
    # Identity survives, or the snapshot cannot be attributed to a run or a receipt.
    assert payload["step"] == 10 and payload["tokenizer"]["sha256"] == "abc"
    assert snap.stat().st_size < store.latest.stat().st_size


def test_weights_only_is_priced_separately(tmp_path: Path):
    """la-700m: 8.41 GB full vs 2.80 GB weights-only, times the retention count."""
    full, weights = 8_410_000_000, 2_800_000_000
    assert Retention(every=2000, keep=3).bytes_for(full, weights) == 25_230_000_000
    assert Retention(every=2000, keep=3, weights_only=True).bytes_for(full, weights) \
        == 8_400_000_000


def test_latest_is_never_trimmed_so_the_current_run_still_resumes(tmp_path: Path):
    import torch

    store = CheckpointStore(tmp_path, Retention(every=10, weights_only=True))
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}, "optimizer": {"state": {}}, "step": 10}, store.latest)
    store.retain(10)
    assert "optimizer" in torch.load(store.latest, map_location="cpu", weights_only=False)
