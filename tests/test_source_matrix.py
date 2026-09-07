"""Per-dataset loss curves: one micro-batch per source, attribution exact rather than estimated.

pt-big was packed merged, so its manifest names one source - `file:.jsonl` - and no amount of
reading its curve says whether the loss came from FineWeb-Edu, Cosmopedia or Python. Reweighting
meant rebuilding the corpus. These assertions are the properties that make the mixture a runtime
decision instead.
"""

import random
from dataclasses import dataclass

import pytest
import torch

from openlocalagent.stages import Stage
from openlocalagent.train.curve import LossCurve, read_curve
from openlocalagent.train.source_matrix import (
    Source,
    SourceMatrix,
    SourceStats,
    Weighting,
)


@dataclass
class FakeShards:
    """Stands in for PackedShardDataset: only sample_batch is used."""

    token: int

    def sample_batch(self, batch_size, rng, device):
        x = torch.full((batch_size, 4), self.token, dtype=torch.long)
        return x, x


def matrix_of(*specs: tuple[str, float, int]) -> SourceMatrix:
    return SourceMatrix([Source(name, FakeShards(i), weight, tokens)
                         for i, (name, weight, tokens) in enumerate(specs)])


def test_weighting_by_sqrt_does_not_erase_the_small_corpus():
    """Equal weighting has been measured to hurt: a 30x oversample cost xlam 14.5 points."""
    big, small = Weighting.SQRT_TOKENS.share(4_000_000_000, 1.0), \
        Weighting.SQRT_TOKENS.share(400_000_000, 1.0)
    assert big / small == pytest.approx(10 ** 0.5, rel=1e-6)
    # Raw token weighting would have been the full 10x.
    assert Weighting.TOKENS.share(4_000_000_000, 1.0) / \
        Weighting.TOKENS.share(400_000_000, 1.0) == pytest.approx(10.0)


def test_explicit_weighting_ignores_size():
    assert Weighting.EXPLICIT.share(4_000_000_000, 2.5) == 2.5


def test_a_draw_carries_its_source_name():
    """The point of the whole module: a batch that does not know where it came from is useless."""
    matrix = matrix_of(("fineweb", 1.0, 100), ("wikipedia", 1.0, 100))
    draw = matrix.draw(2, random.Random(0), "cpu")
    assert draw.name in matrix.names
    assert draw.inputs.shape == (2, 4)


def test_shares_follow_weights():
    matrix = matrix_of(("big", 9.0, 900), ("small", 1.0, 100))
    rng = random.Random(0)
    drawn = [matrix.draw(1, rng, "cpu").name for _ in range(2000)]
    assert 0.85 < drawn.count("big") / len(drawn) < 0.95


def test_composition_is_reportable():
    matrix = matrix_of(("a", 3.0, 300), ("b", 1.0, 100))
    composition = matrix.composition()
    assert composition["a"]["share"] == pytest.approx(0.75)
    assert composition["b"]["tokens"] == 100


def test_empty_matrix_is_refused():
    with pytest.raises(ValueError):
        SourceMatrix([])


def test_missing_source_directory_is_a_hard_error(tmp_path):
    """Training on four corpora when the config names five is invisible in a loss curve."""
    with pytest.raises(FileNotFoundError, match="build_source_shards"):
        SourceMatrix.from_config({"sources": [{"name": "ghost", "path": str(tmp_path / "nope")}]})


def test_stats_weight_loss_by_tokens():
    """A source drawn with shorter rows must not be over-counted when means are compared."""
    stats = SourceStats()
    stats.add(loss=2.0, tokens=1000)
    stats.add(loss=4.0, tokens=3000)
    assert stats.mean_loss == pytest.approx(3.5)      # not the unweighted 3.0
    assert stats.draws == 2 and stats.tokens == 4000


def test_empty_stats_report_no_loss():
    assert SourceStats().mean_loss is None
    assert SourceStats().as_record()["loss"] is None


def test_curve_records_one_loss_per_dataset(tmp_path):
    """The deliverable: a curve you can plot one line per corpus from."""
    curve = LossCurve(tmp_path, Stage.PRETRAIN)
    stats = {"fineweb": SourceStats(), "wikipedia": SourceStats()}
    stats["fineweb"].add(2.4, 1000)
    stats["wikipedia"].add(1.9, 500)
    curve.record(step=0, loss=2.23, learning_rate=3e-4, sources=stats)

    point = read_curve(tmp_path)[0]
    assert point.source_losses() == {"fineweb": 2.4, "wikipedia": 1.9}
    assert point.sources["fineweb"]["tokens"] == 1000


def test_single_corpus_curve_keeps_its_old_shape(tmp_path):
    """Existing readers must not have to learn a new field for runs that have no mixture."""
    curve = LossCurve(tmp_path, Stage.PRETRAIN)
    curve.record(step=0, loss=2.5)
    assert read_curve(tmp_path)[0].sources == {}
    assert "sources" not in (tmp_path / "curve.jsonl").read_text()


# --- the three stages attribute loss differently, and the difference is the point ---


def test_midtrain_style_attribution_is_exact():
    """One micro-batch, one source: the loss the trainer computed IS the source's loss."""
    stats = {"toucan": SourceStats(), "device": SourceStats()}
    stats["toucan"].add(2.10, 4096)      # a whole micro-batch from one corpus
    stats["device"].add(1.75, 4096)
    assert stats["toucan"].mean_loss == pytest.approx(2.10)
    assert stats["device"].mean_loss == pytest.approx(1.75)


def test_posttrain_style_attribution_needs_per_row_loss():
    """An sft micro-batch mixes sources, so splitting its mean by row share would be a guess.

    Two rows, one from each split, with very different losses. Row-share weighting of the batch
    mean gives both the same number; per-row summation recovers the truth. This is why sft reduces
    cross-entropy per row rather than reusing the scalar the model already returned.
    """
    row_losses = [(4.0, 2), (1.0, 2)]                 # (summed loss, counted tokens)
    sources = ["androidcontrol", "xlam"]
    stats: dict[str, SourceStats] = {}
    for (total, counted), source in zip(row_losses, sources):
        stats.setdefault(source, SourceStats()).add(total / counted, counted)

    assert stats["androidcontrol"].mean_loss == pytest.approx(2.0)
    assert stats["xlam"].mean_loss == pytest.approx(0.5)
    # The estimate this replaces would have given both the batch mean, 1.25.
    batch_mean = sum(t for t, _ in row_losses) / sum(c for _, c in row_losses)
    assert batch_mean == pytest.approx(1.25)
    assert stats["androidcontrol"].mean_loss != pytest.approx(batch_mean)


def test_rows_with_no_supervised_tokens_are_not_counted():
    """A fully masked row contributes no loss and must not divide by zero."""
    stats: dict[str, SourceStats] = {}
    for total, counted, source in [(0.0, 0, "toolbench"), (3.0, 3, "toolbench")]:
        if counted:
            stats.setdefault(source, SourceStats()).add(total / counted, counted)
    assert stats["toolbench"].draws == 1 and stats["toolbench"].mean_loss == pytest.approx(1.0)


def test_source_label_strips_fit_tag_and_path():
    from openlocalagent.train.source_matrix import source_label

    assert source_label("data/public/fit2048-2f49a644-xlam.jsonl") == "xlam"
    assert source_label("data/public/fit2048-2f49a644-merged-v2.jsonl") == "merged-v2"
    assert source_label("data/public/toucan-train.jsonl") == "toucan-train"
    assert source_label("finemath") == "finemath"          # already a name
