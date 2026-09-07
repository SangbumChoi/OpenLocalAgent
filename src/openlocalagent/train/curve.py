"""Per-step training curve, written as it happens.

Stages previously kept their loss history in memory and wrote it into `metrics.json` only when
the run finished, so a run that was still going - or that died - had no curve at all. This appends
one JSON line per logged step instead, which makes the curve readable while training is live (the
demo plots it straight from here) and survives a crash.

Append-only by design: a resumed run continues the same file, and re-reading it gives the whole
history including the steps that preceded the resume.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openlocalagent.stages import Stage

CURVE_FILENAME = "curve.jsonl"

_KNOWN_FIELDS = frozenset(
    {"run", "stage", "step", "loss", "elapsed_seconds",
     "learning_rate", "loss_tokens", "validation_loss", "sources"}
)


@dataclass(frozen=True)
class CurvePoint:
    """One logged step of a training run."""

    run: str
    stage: Stage
    step: int
    loss: float
    elapsed_seconds: float
    learning_rate: float | None = None
    loss_tokens: int | None = None
    validation_loss: float | None = None
    #: Per-source loss for this step: {source name: {draws, tokens, loss}}. Empty for a run with
    #: one corpus. This is the field that makes "which dataset is the loss coming from" answerable
    #: from the curve alone, instead of requiring the mixture to be rebuilt to find out.
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Stage-specific series (router aux loss, grad norm, ...) kept out of the closed fields.
    extra: dict[str, Any] = field(default_factory=dict)

    def source_losses(self) -> dict[str, float]:
        """Just the per-source means, for plotting one line per dataset."""
        return {name: rec["loss"] for name, rec in self.sources.items()
                if rec.get("loss") is not None}

    @classmethod
    def from_record(cls, raw: dict[str, Any]) -> CurvePoint:
        return cls(
            run=raw["run"],
            stage=Stage.parse(raw["stage"]),
            step=raw["step"],
            loss=raw["loss"],
            elapsed_seconds=raw["elapsed_seconds"],
            learning_rate=raw.get("learning_rate"),
            loss_tokens=raw.get("loss_tokens"),
            validation_loss=raw.get("validation_loss"),
            sources=raw.get("sources") or {},
            extra={key: value for key, value in raw.items() if key not in _KNOWN_FIELDS},
        )


class LossCurve:
    """Appends `CurvePoint` records to `<out_dir>/curve.jsonl`."""

    def __init__(self, out_dir: str | os.PathLike[str], stage: Stage, run_name: str | None = None):
        self.path = Path(out_dir) / CURVE_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.run_name = run_name or Path(out_dir).name
        self._started = time.time()

    def record(
        self,
        *,
        step: int,
        loss: float,
        learning_rate: float | None = None,
        loss_tokens: int | None = None,
        validation_loss: float | None = None,
        sources: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        record: dict[str, Any] = {
            "run": self.run_name,
            "stage": self.stage.value,
            "step": step,
            "loss": round(float(loss), 6),
            "elapsed_seconds": round(time.time() - self._started, 3),
        }
        if learning_rate is not None:
            record["learning_rate"] = float(learning_rate)
        if loss_tokens is not None:
            record["loss_tokens"] = int(loss_tokens)
        if validation_loss is not None:
            record["validation_loss"] = round(float(validation_loss), 6)
        if sources:
            # A per-source block is written only when the run actually has a mixture, so a
            # single-corpus curve keeps the shape every existing reader already handles.
            record["sources"] = {
                name: (stat.as_record() if hasattr(stat, "as_record") else stat)
                for name, stat in sources.items()
            }
        record.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def read_curve(out_dir: str | os.PathLike[str]) -> list[CurvePoint]:
    """Every point written for a run, oldest first. A missing file reads as no history."""
    path = Path(out_dir) / CURVE_FILENAME
    if not path.exists():
        return []
    return [
        CurvePoint.from_record(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
