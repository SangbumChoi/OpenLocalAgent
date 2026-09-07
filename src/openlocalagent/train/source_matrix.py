"""Draw pretraining batches from many corpora at once, and remember which one each came from.

`pt-big` was packed as one merged file, so its manifest records exactly one source -
``file:.jsonl`` - and the loss it produces cannot be attributed to FineWeb-Edu, Cosmopedia or
permissive Python. Reweighting the mixture meant rebuilding the corpus. Midtrain never had that
problem because it draws a micro-batch per source; this gives pretrain the same property.

One micro-batch comes from one source. That is the whole design: attribution is exact rather than
estimated, the per-source loss the trainer already computes is the number we want, and no batch
mixes two corpora in a way that would have to be untangled afterwards.

Weights are sampling probabilities, normalised. `sqrt` weighting is available because equal
weighting has already been measured to hurt here: giving web control's 3,028 rows the same share
as GUI control's 91,636 oversampled it roughly 30x and cost xlam 14.5 points.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch


class Weighting(StrEnum):
    """How a source's share is derived from its size."""

    #: Use the weights exactly as configured.
    EXPLICIT = "explicit"
    #: Proportional to token count - the biggest corpus dominates.
    TOKENS = "tokens"
    #: Proportional to sqrt(tokens): large corpora still lead, small ones are not erased.
    SQRT_TOKENS = "sqrt_tokens"

    def share(self, tokens: int, configured: float) -> float:
        match self:
            case Weighting.EXPLICIT:
                return configured
            case Weighting.TOKENS:
                return float(tokens)
            case Weighting.SQRT_TOKENS:
                return math.sqrt(max(tokens, 1))


@dataclass(frozen=True, slots=True)
class SourceDraw:
    """One micro-batch and the corpus it came from."""

    name: str
    inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(slots=True)
class SourceStats:
    """Running per-source totals for one logged step.

    Loss is accumulated weighted by tokens rather than by batch, so a source that happens to be
    drawn with shorter rows is not over-counted when the means are compared.
    """

    draws: int = 0
    loss_sum: float = 0.0
    tokens: int = 0

    def add(self, loss: float, tokens: int) -> None:
        self.draws += 1
        self.loss_sum += loss * tokens
        self.tokens += tokens

    @property
    def mean_loss(self) -> float | None:
        return self.loss_sum / self.tokens if self.tokens else None

    def as_record(self) -> dict[str, float | int]:
        return {"draws": self.draws, "tokens": self.tokens,
                "loss": round(self.mean_loss, 6) if self.mean_loss is not None else None}


@dataclass(frozen=True, slots=True)
class Source:
    """One corpus in the mixture."""

    name: str
    dataset: object          # PackedShardDataset; typed loosely to keep this import-light
    weight: float
    tokens: int

    def sample(self, batch_size: int, rng: random.Random, device) -> SourceDraw:
        inputs, targets = self.dataset.sample_batch(batch_size, rng, device)
        return SourceDraw(self.name, inputs, targets)


class SourceMatrix:
    """A weighted set of packed corpora, sampled one micro-batch at a time.

    Deliberately not a torch Dataset: the trainer needs the source name alongside the tensors, and
    wrapping that in a collate function would hide exactly the thing this exists to expose.
    """

    def __init__(self, sources: list[Source]):
        if not sources:
            raise ValueError("a source matrix needs at least one source")
        total = sum(source.weight for source in sources)
        if total <= 0:
            raise ValueError("source weights must sum to something positive")
        self.sources = sources
        self._shares = [source.weight / total for source in sources]

    @classmethod
    def from_config(cls, spec: dict, split: str = "train") -> SourceMatrix:
        """Build from a config's ``data.sources`` block.

            data:
              weighting: sqrt_tokens
              sources:
                - {name: fineweb, path: data/shards/src/fineweb}
                - {name: wikipedia, path: data/shards/src/wikipedia, weight: 2.0}

        A source whose directory is missing is a hard error: silently training on four corpora
        when the config names five is a difference nobody would notice in a loss curve.
        """
        from openlocalagent.data.corpus.pretrain_corpus import PackedShardDataset

        weighting = Weighting(spec.get("weighting", Weighting.SQRT_TOKENS))
        sources = []
        for entry in spec["sources"]:
            path = Path(entry["path"])
            if not (path / "manifest.json").is_file():
                raise FileNotFoundError(
                    f"source {entry['name']!r} has no packed manifest at {path}; "
                    "build it with scripts/build_source_shards.py"
                )
            dataset = PackedShardDataset(path, split)
            tokens = sum(int(v) for v in dataset.manifest.get("source_token_counts", {}).values())
            sources.append(Source(
                name=str(entry["name"]),
                dataset=dataset,
                weight=weighting.share(tokens, float(entry.get("weight", 1.0))),
                tokens=tokens,
            ))
        return cls(sources)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(source.name for source in self.sources)

    def composition(self) -> dict[str, dict[str, float | int]]:
        """What the mixture actually is, for the run's metadata and for a human to check."""
        return {
            source.name: {"tokens": source.tokens, "share": round(share, 6)}
            for source, share in zip(self.sources, self._shares)
        }

    def draw(self, batch_size: int, rng: random.Random, device) -> SourceDraw:
        """One micro-batch from one source, chosen by weight."""
        source = rng.choices(self.sources, weights=self._shares, k=1)[0]
        return source.sample(batch_size, rng, device)

    def fresh_stats(self) -> dict[str, SourceStats]:
        """Zeroed counters in a deterministic order, so a curve's keys never reorder."""
        return {name: SourceStats() for name in self.names}


def source_label(source: object) -> str:
    """A curve key a person can read: `xlam`, not `data/public/fit2048-2f49a644-xlam.jsonl`.

    sft sources are file paths, and context-fitted files carry a budget and tokenizer tag in
    their name. Both are provenance worth keeping in the run metadata and noise in a plot legend.
    A source that is already a bare name passes through unchanged.
    """
    import re
    from pathlib import Path

    text = str(source)
    if "/" not in text and not text.endswith(".jsonl"):
        return text
    stem = Path(text).stem
    return re.sub(r"^fit\d+-[0-9a-f]{8}-", "", stem)
