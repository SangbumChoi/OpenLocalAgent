#!/usr/bin/env python3
"""Run an isolated bounded pretraining, SFT, or RL memory/throughput preflight."""

from __future__ import annotations

import argparse
import json

from localagent.train.update_preflight import run_one_update_training_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="production pretraining, SFT, or RL YAML config")
    parser.add_argument(
        "--work-dir",
        required=True,
        help="new directory for the derived config, checkpoint, and metrics",
    )
    parser.add_argument("--out", required=True, help="new self-hashed JSON receipt path")
    parser.add_argument(
        "--device",
        help="optional derived runtime.device override; defaults to the production request",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run_one_update_training_preflight(
        args.config,
        work_dir=args.work_dir,
        receipt_path=args.out,
        device=args.device,
    )
    stage = receipt["effective"]["contract"]["stage"]
    summary = {
        "stage": stage,
        "status": receipt["status"],
        "receipt": args.out,
        "receipt_self_sha256": receipt["receipt_self_sha256"],
        "resolved_device": receipt["metrics"]["execution"]["resolved_device"],
        "wall_seconds": receipt["measurement"]["wall_seconds"],
        "peak_process_rss_bytes": receipt["measurement"]["memory"][
            "peak_process_rss_bytes"
        ],
        "peak_accelerator_bytes": max(
            receipt["measurement"]["memory"]["peak_cuda_reserved_bytes"],
            receipt["measurement"]["memory"]["peak_mps_driver_allocated_bytes"],
        ),
    }
    if stage == "rl":
        accounting = receipt["metrics"]["rl_accounting"]
        observability = accounting["rollout_observability"]
        parsing = observability["parsing"]
        attempted = accounting["attempted_rollouts"]
        transition = receipt["measurement"]["policy_transition"]
        summary.update(
            {
                "realized_optimizer_updates": accounting["realized_optimizer_updates"],
                "learning_rate_history": transition["actual_learning_rates"],
                "nonzero_learning_rate_executed": transition[
                    "nonzero_learning_rate_executed"
                ],
                "changed_policy_tensors": transition[
                    "changed_model_parameter_count"
                ],
                "reward_unique_values": observability["reward"]["unique_values"],
                "parser_format_valid_rate": (
                    parsing["parser_format_valid_rollouts"] / attempted
                    if attempted
                    else None
                ),
                "truncated_rollouts": observability["truncation"][
                    "truncated_rollouts"
                ],
                "generated_tokens": observability["tokens"]["generated_tokens"],
            }
        )
    elif stage == "sft":
        contract = receipt["effective"]["contract"]
        summary.update(
            {
                "realized_optimizer_updates": contract["realized_optimizer_updates"],
                "execution_optimizer_update_limit": contract[
                    "execution_optimizer_update_limit"
                ],
                "micro_batch_size": contract["micro_batch_size"],
                "grad_accum_steps": contract["grad_accum_steps"],
                "effective_batch_size": contract["effective_batch_size"],
                "loss_last": receipt["metrics"]["loss_last"],
                "input_tokens": receipt["metrics"]["token_accounting"]["input_tokens"],
            }
        )
    print(
        json.dumps(
            summary,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
