#!/usr/bin/env python3
"""Train directly from pinned Hugging Face streams without local JSONL or shards."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from openlocalagent.data.corpus.online_stream import OnlineSourceDataset
from openlocalagent.data.hf_corpus import build_mixture_plan
from openlocalagent.env import load_env_file
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer
from openlocalagent.train.pretrain import pretrain
from openlocalagent.train.source_matrix import Source, SourceMatrix


def run(config_path: str | Path) -> dict[str, object]:
    """Run a fresh online experiment; source iterators are never replaced by cached rows."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("runtime", {}).get("resume", False):
        raise ValueError("online stream resume is not exact; start a new run instead")
    load_env_file()
    data = config["data"]
    plan = build_mixture_plan(data["mixture_config"])
    available = {source["name"]: source for source in plan["sources"]}
    names = data.get("sources") or list(available)
    if len(set(names)) != len(names) or set(names) - set(available):
        raise ValueError("data.sources must name unique sources from the mixture config")
    model_config = ModelConfig.from_yaml(config["model_config"])
    model_config.assert_within_budget()
    tokenizer = ByteTokenizer()
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("online runner currently requires a 256-vocab byte model")
    sequence = int(data["seq_len"])
    if sequence > model_config.max_seq_len:
        raise ValueError("data.seq_len exceeds model max_seq_len")
    seed = int(config.get("runtime", {}).get("seed", 2026))
    torch.manual_seed(seed)
    datasets = {
        name: OnlineSourceDataset(
            available[name],
            tokenizer,
            seq_len=sequence,
            seed=seed + index,
            shuffle_buffer=int(data.get("shuffle_buffer", 128)),
            preprocessor=data.get("preprocessors", {}).get(name),
        )
        for index, name in enumerate(names)
    }
    matrix = SourceMatrix(
        [
            Source(name, datasets[name], float(available[name]["weight"]), 0)
            for name in names
        ]
    )
    target_shares = {
        name: details["share"] for name, details in matrix.composition().items()
    }
    source_transports = {
        name: (
            "http_gzip_partial_unverified"
            if (available[name].get("raw_stream") or {}).get("backend") == "hf-jsonl-gzip-v1"
            else "hf_iterable_stream"
        )
        for name in names
    }
    output = Path(config["log"]["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    curve_path = output / "curve.jsonl"
    if curve_path.exists():
        raise FileExistsError(f"online run output already has a curve: {curve_path}")
    runtime = config.get("runtime", {})
    train = config["train"]
    track = config.get("tracking", {})
    wandb_run = None
    if track.get("wandb", False):
        import wandb

        mode = track.get("mode", "online")
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("online W&B requires WANDB_API_KEY in environment or .env")
        wandb_run = wandb.init(
            project=track.get("project", "openlocalagent-pretrain-stream"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            mode=mode,
            config={"model_config": config["model_config"], "target_shares": target_shares},
        )
    started = time.time()

    def on_step(record: dict[str, object]) -> None:
        for name, dataset in datasets.items():
            for key, value in dataset.stats().items():
                record[f"stream/{name}/{key}"] = value
        with curve_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if wandb_run is not None:
            wandb_run.log(record)

    try:
        history, metrics = pretrain(
            LocalAgentLM(model_config),
            datasets[names[0]],
            tokenizer,
            matrix=matrix,
            steps=int(train["steps"]),
            batch_size=int(train["batch_size"]),
            accum_steps=int(train["accum_steps"]),
            seq_len=sequence,
            lr=float(train["lr"]),
            warmup=int(train.get("warmup", 0)),
            device=runtime.get("device", "cpu"),
            seed=seed,
            amp_dtype=getattr(torch, runtime.get("dtype", "float32")),
            checkpoint_path=output / "latest.pt",
            checkpoint_every=int(train.get("checkpoint_every", 0)),
            metrics_callback=on_step,
            return_metrics=True,
            data_metadata={
                "kind": "online_hf_stream",
                "mixture_config": data["mixture_config"],
                "sources": list(names),
                "exact_resume": False,
                "evaluation_holdout": "unfrozen",
                "source_transports": source_transports,
            },
        )
    finally:
        for dataset in datasets.values():
            dataset.close()
        if wandb_run is not None:
            wandb_run.finish()
        # HF streaming can retain cyclic Arrow/file-system objects past generator closure.
        # Collect while Python still owns the GIL, before interpreter finalization.
        gc.collect()
    summary: dict[str, object] = {
        "kind": "online_hf_pretrain_pilot",
        "sources": {name: dataset.stats() for name, dataset in datasets.items()},
        "target_shares": target_shares,
        "source_transports": source_transports,
        "steps": metrics["steps_completed"],
        "initial_loss": history[0],
        "final_loss": history[-1],
        "loss_tokens": metrics["token_accounting"]["loss_tokens"],
        "elapsed_seconds": round(time.time() - started, 2),
        "checkpoint": str(output / "latest.pt"),
        "curve": str(curve_path),
        "wandb_url": getattr(wandb_run, "url", None),
        "evaluation_holdout": "unfrozen; do not use for paper benchmark claims",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain/la-1m-online.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2), flush=True)
    # The HF IterableDataset/Arrow runtime on the tested CUDA host can abort in a native
    # finalizer after every durable output and W&B sync has completed. This opt-in exit applies
    # only on success; exceptions still fail normally. It bypasses native interpreter teardown.
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config.get("runtime", {}).get("skip_native_shutdown", False):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
