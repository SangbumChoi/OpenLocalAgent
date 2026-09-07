"""Retained, resumable checkpoints - one writer for all three stages.

Every stage already saves everything resuming needs: weights, optimizer state, step, the python /
torch / cuda / mps RNG states and the grad scaler. What none of them kept was a way *back*. Each
save overwrote a single ``latest.pt``, so a run directory held exactly one point in time and a
bad write destroyed the run.

That is not hypothetical. Two queue copies once interleaved writes into ``posttrain-la-300m`` and
the whole run had to be deleted, because "the step before the corruption" did not exist anywhere.

Snapshots here are free at write time. ``latest.pt`` is written to a temp file and renamed, which
gives it a *new* inode, so hard-linking a stamped name to the current ``latest.pt`` before the
next save costs no bytes and no copy - the old data simply stays reachable under its own name
once ``latest.pt`` moves on. Disk is spent only by retention, which is bounded and explicit,
because a 700M checkpoint is 7.9 GB and an unbounded policy fills a shared NFS volume.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

#: ``step-000016000.pt`` - fixed width so lexical order is numeric order.
SNAPSHOT = re.compile(r"^step-(\d{9})\.pt$")
LATEST = "latest.pt"


def snapshot_name(step: int) -> str:
    return f"step-{step:09d}.pt"


@dataclass(frozen=True, slots=True)
class Retention:
    """How many points in time a run keeps, and how far apart.

    Defaults keep the feature off, so an existing config's disk use cannot change under it.
    Turning it on is a config edit, and the cost is stated in tiers of checkpoint size rather
    than discovered when the volume fills.
    """

    #: Link a stamped snapshot every N steps. 0 disables snapshots entirely.
    every: int = 0
    #: Keep at most this many stamped snapshots, newest first. 0 means keep all.
    keep: int = 3
    #: Drop the optimizer from snapshots. Measured on la-700m: state_dict is 2.80 GB and the
    #: AdamW state is 5.61 GB, so this is a 3x saving - two moments per parameter at fp32 is
    #: twice the weights. The trade is exact: a weights-only snapshot can re-score a point but
    #: cannot resume training from it, because Adam's moments are not recoverable. latest.pt is
    #: never trimmed, so resuming the *current* run is unaffected either way.
    weights_only: bool = False

    @classmethod
    def from_log_config(cls, log_cfg: dict) -> Retention:
        """Read `log.snapshot_every` / `log.snapshot_keep`, defaulting to off."""
        return cls(every=int(log_cfg.get("snapshot_every", 0)),
                   keep=int(log_cfg.get("snapshot_keep", 3)),
                   weights_only=bool(log_cfg.get("snapshot_weights_only", False)))

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def due(self, step: int) -> bool:
        """Whether `step` (1-based count of completed steps) lands on a snapshot boundary."""
        return self.enabled and step > 0 and step % self.every == 0

    def bytes_for(self, checkpoint_bytes: int, weights_bytes: int | None = None) -> int:
        """Worst-case disk this policy adds, so a config can be costed before it runs."""
        if not self.enabled:
            return 0
        per = weights_bytes if (self.weights_only and weights_bytes is not None) \
            else checkpoint_bytes
        return per * max(self.keep, 1)


@dataclass(frozen=True, slots=True)
class CheckpointStore:
    """A run directory's points in time.

    Deliberately does not know how to *build* a payload - each stage owns that, and the payload
    already carries optimizer, step, RNG and scaler state. This owns only which files exist.
    """

    out_dir: Path
    retention: Retention = Retention()

    @property
    def latest(self) -> Path:
        return self.out_dir / LATEST

    def snapshots(self) -> list[Path]:
        """Stamped snapshots, oldest first."""
        return sorted(
            (p for p in self.out_dir.glob("step-*.pt") if SNAPSHOT.match(p.name)),
            key=lambda p: int(SNAPSHOT.match(p.name).group(1)),
        )

    def step_of(self, path: Path) -> int:
        match = SNAPSHOT.match(path.name)
        if match is None:
            raise ValueError(f"{path.name} is not a snapshot")
        return int(match.group(1))

    def retain(self, step: int) -> Path | None:
        """Hard-link the current ``latest.pt`` to a stamped name, then prune.

        Called *after* the stage has written latest.pt for this step. Returns the snapshot path,
        or None when the policy says nothing is due. A hard link costs no bytes: the next save
        replaces latest.pt with a new inode and this name keeps the old one alive.
        """
        if not self.retention.due(step) or not self.latest.exists():
            return None
        target = self.out_dir / snapshot_name(step)
        if target.exists():
            return target
        if self.retention.weights_only:
            return self._write_weights_only(target)
        try:
            os.link(self.latest, target)
        except OSError:
            # Cross-device or a filesystem without hard links: a copy still preserves the point
            # in time, which is the whole purpose. Pay the bytes rather than lose the snapshot.
            import shutil

            temporary = target.with_suffix(".pt.tmp")
            shutil.copy2(self.latest, temporary)
            temporary.replace(target)
        self.prune()
        return target

    def prune(self) -> list[Path]:
        """Drop the oldest snapshots beyond the retention count; return what was removed."""
        if self.retention.keep <= 0:
            return []
        existing = self.snapshots()
        removed = []
        for path in existing[: max(0, len(existing) - self.retention.keep)]:
            path.unlink(missing_ok=True)
            removed.append(path)
        return removed

    def resume_from(self, step: int | None = None) -> Path | None:
        """The file to resume from: ``latest.pt``, or the newest snapshot at or before `step`.

        Passing a step is the recovery path - after a corrupt or unattributable ``latest.pt``,
        a run can restart from a point that is known good instead of from zero.
        """
        if step is None:
            return self.latest if self.latest.exists() else None
        candidates = [p for p in self.snapshots() if self.step_of(p) <= step]
        return candidates[-1] if candidates else None

    def _write_weights_only(self, target: Path) -> Path:
        """Snapshot without the optimizer: re-scoreable, not resumable.

        Everything that identifies the point in time is kept - step, tokenizer and data lineage,
        config - so the artifact can still be attributed to a run and a receipt. Only the two
        Adam moments are dropped, and the payload says so rather than looking like a full one.
        """
        import torch

        payload = torch.load(self.latest, map_location="cpu", weights_only=False)
        payload.pop("optimizer", None)
        payload["snapshot_kind"] = "weights_only"
        payload["resumable"] = False
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        self.prune()
        return target


# --- safetensors payloads -----------------------------------------------------------------
#
# A training checkpoint is a tree of tensors mixed with scalars: cfg, step and lineage sit beside
# state_dict, the optimizer's two moments per parameter, and four kinds of RNG state. safetensors
# stores a FLAT dict of tensors plus a str->str metadata header and nothing else, so the tree has
# to be split rather than dumped.
#
# The split is generic: walk the payload, lift every tensor into a flat dict keyed by its dotted
# path, and leave a typed placeholder behind. What remains is JSON, and it round-trips exactly -
# including the tuples that RNG state arrives as, which JSON would otherwise silently turn into
# lists and torch would then refuse.
#
# Why bother, when .pt already works: a .pt is a pickle, so loading one executes whatever it
# contains. safetensors cannot execute anything, is memory-mappable, and is what every other tool
# in the ecosystem reads. The optimizer moments are tensors like any other, so "resumable" and
# "safetensors" were never actually in conflict - only the scalars needed somewhere to go.

TENSOR_MARK = "__tensor__:"
TUPLE_MARK = "__tuple__"


def _split(value, path: str, tensors: dict):
    """Replace tensors with placeholders, recording them under their dotted path."""
    import torch

    if torch.is_tensor(value):
        # safetensors rejects shared storage and non-contiguous views alike.
        tensors[path] = value.detach().contiguous().clone()
        return TENSOR_MARK + path
    if isinstance(value, dict):
        return {str(k): _split(v, f"{path}.{k}" if path else str(k), tensors)
                for k, v in value.items()}
    if isinstance(value, tuple):
        return {TUPLE_MARK: [_split(v, f"{path}.{i}", tensors) for i, v in enumerate(value)]}
    if isinstance(value, list):
        return [_split(v, f"{path}.{i}", tensors) for i, v in enumerate(value)]
    return value


def _join(value, tensors: dict):
    """Inverse of _split: put the tensors back and restore tuple-ness."""
    if isinstance(value, str) and value.startswith(TENSOR_MARK):
        return tensors[value[len(TENSOR_MARK):]]
    if isinstance(value, dict):
        if set(value) == {TUPLE_MARK}:
            return tuple(_join(v, tensors) for v in value[TUPLE_MARK])
        return {k: _join(v, tensors) for k, v in value.items()}
    if isinstance(value, list):
        return [_join(v, tensors) for v in value]
    return value


def save_safetensors(payload: dict, path: Path) -> Path:
    """Write a full training payload as safetensors + a JSON skeleton in the metadata header."""
    import json

    from safetensors.torch import save_file

    tensors: dict = {}
    skeleton = _split(payload, "", tensors)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    # Integer keys (optimizer.state is keyed by param index) do not survive JSON as keys; the
    # skeleton keeps them as strings and _restore_int_keys puts them back on load.
    save_file(tensors, str(temporary), metadata={"localagent_skeleton": json.dumps(skeleton)})
    temporary.replace(path)
    return path


def load_safetensors(path: Path, map_location: str = "cpu") -> dict:
    """Read back a payload written by save_safetensors, tensors and scalars alike."""
    import json

    from safetensors import safe_open

    tensors: dict = {}
    with safe_open(str(path), framework="pt", device=map_location) as handle:
        skeleton = json.loads(handle.metadata()["localagent_skeleton"])
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    payload = _join(skeleton, tensors)
    return _restore_int_keys(payload)


def _restore_int_keys(payload: dict) -> dict:
    """torch's optimizer state is keyed by integer param index; JSON made those strings.

    Loading with string keys does not fail - it silently produces an optimizer whose state
    matches no parameter, so the run resumes with the moments quietly discarded.
    """
    optimizer = payload.get("optimizer")
    if isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict):
        optimizer["state"] = {
            (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): v
            for k, v in optimizer["state"].items()
        }
    return payload
