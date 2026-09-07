"""The per-step training curve every stage writes."""

from __future__ import annotations

import json

from openlocalagent.stages import Stage
from openlocalagent.train.curve import CURVE_FILENAME, LossCurve, read_curve


def test_records_are_appended_one_per_line(tmp_path) -> None:
    curve = LossCurve(tmp_path, Stage.PRETRAIN, run_name="la-93m-pretrain")
    curve.record(step=0, loss=6.5, learning_rate=1e-4, loss_tokens=2048)
    curve.record(step=1, loss=6.1, learning_rate=2e-4, loss_tokens=4096)

    lines = (tmp_path / CURVE_FILENAME).read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 0
    assert json.loads(lines[1])["loss"] == 6.1


def test_read_curve_types_the_records(tmp_path) -> None:
    curve = LossCurve(tmp_path, Stage.MIDTRAIN, run_name="la-45m-midtrain")
    curve.record(step=7, loss=2.25, validation_loss=2.4, source_draws=12)

    points = read_curve(tmp_path)
    assert len(points) == 1
    point = points[0]
    assert point.run == "la-45m-midtrain"
    assert point.stage is Stage.MIDTRAIN
    assert point.step == 7
    assert point.loss == 2.25
    assert point.validation_loss == 2.4
    # stage-specific series stay available without widening the closed fields
    assert point.extra["source_draws"] == 12
    assert point.elapsed_seconds >= 0.0


def test_a_resumed_run_continues_the_same_file(tmp_path) -> None:
    LossCurve(tmp_path, Stage.POSTTRAIN).record(step=0, loss=1.0)
    LossCurve(tmp_path, Stage.POSTTRAIN).record(step=1, loss=0.9)

    assert [point.step for point in read_curve(tmp_path)] == [0, 1]


def test_a_run_with_no_curve_yet_reads_as_empty(tmp_path) -> None:
    assert read_curve(tmp_path) == []
