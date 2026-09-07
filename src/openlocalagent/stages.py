"""The pipeline's stage vocabulary.

Training is three stages, and each one answers a different question:

    pretrain   can it model language at all?          text corpora, loss in bits/byte
    midtrain   does it know what an agent does?       agentic trajectories, tool catalogs
    posttrain  does it emit the exact call asked for? supervised fine-tuning on the target format

distill, rl and eval hang off that spine rather than extending it.

The stage was previously a bare string repeated across the CLI, the flow DAG, the config `stage:`
key and the training modules, with the runnable entry points held in a separate dict keyed by that
string - so adding a stage meant editing several places that could disagree. Everything a stage
knows now lives on the member.
"""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    PRETRAIN = "pretrain"
    MIDTRAIN = "midtrain"
    POSTTRAIN = "posttrain"
    DISTILL = "distill"
    RL = "rl"
    EVAL = "eval"

    @classmethod
    def parse(cls, text: str) -> Stage:
        """Resolve a stage name, accepting `sft` as the older spelling of `posttrain`."""
        if text == _LEGACY_POSTTRAIN_SPELLING:
            return cls.POSTTRAIN
        try:
            return cls(text)
        except ValueError:
            known = ", ".join(stage.value for stage in cls)
            raise SystemExit(f"unknown stage {text!r}. known: {known}") from None

    @property
    def entry_point(self) -> tuple[str, str]:
        """Module and callable that runs this stage."""
        match self:
            case Stage.PRETRAIN:
                return ("openlocalagent.train.pretrain", "run")
            case Stage.MIDTRAIN:
                return ("openlocalagent.train.midtrain", "run")
            case Stage.POSTTRAIN:
                return ("openlocalagent.train.sft", "run")
            case Stage.DISTILL:
                return ("openlocalagent.train.distill", "run")
            case Stage.RL:
                return ("openlocalagent.train.rl", "run")
            case Stage.EVAL:
                return ("openlocalagent.eval.harness", "run")

    @property
    def supports_exact_resume(self) -> bool:
        """Whether the stage can resume its own `latest.pt` step-for-step."""
        match self:
            case Stage.PRETRAIN | Stage.MIDTRAIN | Stage.POSTTRAIN | Stage.RL:
                return True
            case Stage.DISTILL | Stage.EVAL:
                return False

    @property
    def checkpoint_stage(self) -> str:
        """The spelling this stage writes into checkpoint and lineage metadata.

        Post-training still writes `sft` there. Existing checkpoints on the training box carry
        that value and the resume/lineage checks compare against it, so the artifact format is
        deliberately left behind the vocabulary rather than invalidating them.
        """
        if self is Stage.POSTTRAIN:
            return _LEGACY_POSTTRAIN_SPELLING
        return self.value


_LEGACY_POSTTRAIN_SPELLING = "sft"

#: The training spine, in the order a run walks it.
TRAINING_STAGES = (Stage.PRETRAIN, Stage.MIDTRAIN, Stage.POSTTRAIN)
