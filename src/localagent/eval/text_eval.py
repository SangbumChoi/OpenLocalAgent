"""Text-generation eval (Phase 5): perplexity + a few small task accuracies."""

from __future__ import annotations


def perplexity(model, jsonl_path: str) -> float:
    raise NotImplementedError("TODO(phase-5): mean NLL over a held-out text set")


def task_accuracy(model, task: str) -> float:
    raise NotImplementedError("TODO(phase-5): small ARC/GSM-style multiple-choice accuracy")
