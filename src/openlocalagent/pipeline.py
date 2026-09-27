"""Runs one stage of the pipeline.

    pretrain -> midtrain -> posttrain -> distill -> eval -> export
                          ^                       |
                          |      data flywheel    |
                          +------- serve <--------+

`scripts/speedrun.sh` is the one-command, toy-scale, end-to-end entry point.
"""

from __future__ import annotations

import importlib

from openlocalagent.stages import Stage


def _entry(stage: Stage):
    module_name, fn_name = stage.entry_point
    return getattr(importlib.import_module(module_name), fn_name)


def run_stage(stage: str | Stage, config_path: str) -> None:
    """Run a stage from its configured starting point."""
    _entry(Stage.parse(stage))(config_path)


def resume_stage(
    stage: str | Stage,
    config_path: str,
    *,
    git_receipt: str | None = None,
) -> None:
    """Resume a stage's existing `latest.pt` step for step."""
    stage = Stage.parse(stage)
    if not stage.supports_exact_resume:
        raise SystemExit(f"stage '{stage}' does not support exact resume")
    if git_receipt is None:
        _entry(stage)(config_path, resume=True)
        return
    if stage is not Stage.POSTTRAIN:
        raise SystemExit("--resume-git-receipt currently supports only the posttrain stage")
    _entry(stage)(config_path, resume=True, resume_git_receipt=git_receipt)
