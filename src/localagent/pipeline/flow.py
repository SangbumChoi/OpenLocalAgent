"""Pipeline orchestration (Phase 0 stub -> grows with phases).

Wires stages into a runnable DAG with artifact tracking:

    pretrain -> midtrain -> sft -> distill -> eval -> export
                          ^                  |
                          |   data flywheel  |
                          +----- serve <-----+

`scripts/speedrun.sh` is the one-command, toy-scale, end-to-end entry point.
"""

from __future__ import annotations

# Stage name -> (module, callable). Kept declarative so the CLI/speedrun can run any sub-DAG.
STAGES = {
    "pretrain": ("localagent.train.pretrain", "run"),
    "midtrain": ("localagent.train.midtrain", "run"),
    "sft": ("localagent.train.sft", "run"),
    "distill": ("localagent.train.distill", "run"),
    "rl": ("localagent.train.rl", "run"),
    "eval": ("localagent.eval.harness", "run"),
}


def run_stage(
    stage: str,
    config_path: str,
    *,
    resume: bool = False,
    resume_git_receipt: str | None = None,
) -> None:
    import importlib

    if stage not in STAGES:
        raise SystemExit(f"unknown stage '{stage}'. known: {', '.join(STAGES)}")
    if resume and stage not in {"pretrain", "midtrain", "sft", "rl"}:
        raise SystemExit(f"stage '{stage}' does not support exact resume")
    if resume_git_receipt is not None and not resume:
        raise SystemExit("--resume-git-receipt requires --resume")
    if resume_git_receipt is not None and stage != "sft":
        raise SystemExit("--resume-git-receipt currently supports only the sft stage")
    module_name, fn_name = STAGES[stage]
    fn = getattr(importlib.import_module(module_name), fn_name)
    if resume:
        if resume_git_receipt is not None:
            fn(
                config_path,
                resume=True,
                resume_git_receipt=resume_git_receipt,
            )
        else:
            fn(config_path, resume=True)
    else:
        fn(config_path)
