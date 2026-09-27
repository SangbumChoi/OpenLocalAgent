#!/usr/bin/env python3
"""Run a portable WebGPU architecture/transfer/evaluation campaign.

The campaign is intentionally safe to move between machines.  It benchmarks every selected model
configuration, validates the public evaluation matrix, and optionally analyzes checkpoint pairs.
It never downloads benchmark data, starts an emulator, calls an external account, or silently
trains a model.  Training configs are recorded in the receipt; launch those stages explicitly on
the GPU host after reviewing the split and output paths.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from localagent.data.public_eval_matrix import load_matrix
from localagent.model import ModelConfig
from localagent.train.device import resolve_device

try:  # direct script invocation
    from acquire_hf_sources import acquire
    from benchmark_model_configs import run_benchmark
    from analyze_weight_transfer import analyze as analyze_transfer
except ImportError:  # package-style test/import
    from scripts.acquire_hf_sources import acquire
    from scripts.benchmark_model_configs import run_benchmark
    from scripts.analyze_weight_transfer import analyze as analyze_transfer

CAMPAIGN_CONFIG = Path("configs/experiments/webgpu-realistic-campaign.v1.yaml")


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _self_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _preflight(requested_device: str, dtype: str) -> dict[str, Any]:
    device = resolve_device(requested_device)
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {
        "requested_device": requested_device,
        "resolved_device": str(device),
        "requested_dtype": dtype,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if cuda
        else [],
        "mps_available": mps,
        "adb": _which("adb"),
        "emulator": _which("emulator"),
        "docker": _which("docker"),
        "git": _which("git"),
        "hf_cli": _which("hf"),
        "huggingface_hub_importable": importlib.util.find_spec("huggingface_hub") is not None,
        "wandb_importable": importlib.util.find_spec("wandb") is not None,
        "claim_boundary": "Runtime inventory only; no device or external service was mutated.",
    }


def _which(binary: str) -> str | None:
    result = subprocess.run(["sh", "-lc", f"command -v {binary}"], capture_output=True, text=True)
    path = result.stdout.strip()
    return path or None


def _model_inventory(model_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in model_names:
        path = Path("configs/model") / f"{name}.yaml"
        cfg = ModelConfig.from_yaml(path)
        cfg.assert_within_budget()
        rows.append(
            {
                "name": name,
                "config": _identity(path),
                "parameters": cfg.estimate_params(),
                "active_parameters": cfg.estimate_active_params(),
                "effective_depth": cfg.effective_depth,
                "block_types": cfg.block_types(),
                "vision_enabled": cfg.vision_enabled,
                "sparse_ffn": cfg.sparse_ffn,
                "budget_pass": cfg.estimate_params() < 100_000_000,
            }
        )
    return rows


def _load_campaign_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("kind") != "localagent_webgpu_realistic_campaign":
        raise ValueError("invalid campaign config kind")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported campaign config version")
    return raw


def _wandb_log(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.wandb:
        return {"enabled": False}
    try:
        import wandb
    except ImportError as error:  # pragma: no cover - optional runtime
        raise RuntimeError('install W&B tracking with: pip install -e ".[tracking]"') from error
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            "campaign_config_sha256": payload["campaign_config"]["sha256"],
            "device": payload["preflight"]["resolved_device"],
            "models": [row["name"] for row in payload["model_inventory"]],
            "checkpoint_sha256": payload.get("checkpoint", {}).get("sha256"),
        },
    )
    benchmark = payload.get("architecture_benchmark", {})
    for index, row in enumerate(benchmark.get("results", [])):
        wandb.log(
            {
                "model/parameters": row["parameters"],
                "model/prefill_tok_s": row["prefill_tok_s"],
                "model/cached_decode_tok_s": row["cached_decode_tok_s"],
                "model/weight_bytes": row["weight_bytes_estimate"],
                "model/kv_cache_bytes": row["kv_cache_bytes_estimate"],
                "model/name": row["model"],
            },
            step=index,
        )
    run.summary["campaign_claim_boundary"] = payload["claim_boundary"]
    run.finish()
    return {
        "enabled": True,
        "mode": args.wandb_mode,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "run_id": getattr(run, "id", None),
        "url": getattr(run, "url", None),
    }


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    campaign_path = Path(args.campaign).resolve()
    campaign = _load_campaign_config(campaign_path)
    models = args.models or list(campaign["models"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "kind": "localagent_gpu_realistic_campaign",
        "schema_version": 1,
        "started_at_unix": time.time(),
        "campaign_config": _identity(campaign_path),
        "preflight": _preflight(args.device, args.dtype),
        "model_inventory": _model_inventory(models),
        "training_plan": campaign.get("training_plan", {}),
        "comparisons": campaign.get("comparisons", []),
        "evaluation_policy": campaign.get("evaluation_policy", {}),
        "claim_boundary": (
            "This receipt joins portable architecture/transfer measurements and source-linked matrix "
            "validation. It is not a training-quality, native-mobile, desktop, WebGPU-browser, or "
            "external-account score unless a separate bound receipt is supplied."
        ),
    }
    if not args.skip_matrix:
        matrix = load_matrix(args.matrix)
        payload["public_matrix"] = {
            "source": _identity(Path(args.matrix)),
            "entries": len(matrix["entries"]),
            "train_rows": [row["id"] for row in matrix["entries"] if row["train_policy"] == "train"],
            "families": sorted({row["family"] for row in matrix["entries"]}),
            "local_status": {
                status: sum(row["local_status"] == status for row in matrix["entries"])
                for status in sorted({row["local_status"] for row in matrix["entries"]})
            },
        }
    if args.acquire_hf:
        payload["hf_acquisition"] = acquire(
            config_path=args.hf_config,
            output_dir=args.hf_out,
            source_ids=args.hf_sources,
            dry_run=args.hf_dry_run,
        )
    if not args.skip_benchmark:
        payload["architecture_benchmark"] = run_benchmark(
            models,
            device_name=args.device,
            dtype_name=args.dtype,
            prompt_len=args.prompt_len,
            decode=args.decode,
            repeats=args.repeats,
            uncached=not args.no_uncached,
        )
    if args.transfer:
        transfers: list[dict[str, Any]] = []
        for pair in args.transfer:
            if ":" not in pair:
                raise ValueError("--transfer expects BASE:TARGET")
            base, target = (Path(part) for part in pair.split(":", 1))
            transfers.append(analyze_transfer(base, target))
        payload["weight_transfer"] = transfers
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        payload["checkpoint"] = _identity(checkpoint)
        payload["checkpoint_note"] = (
            "Checkpoint identity only. Run the pinned native/browser evaluators separately and add "
            "their receipts; this campaign does not infer task success from checkpoint presence."
        )
    payload["wandb"] = _wandb_log(payload, args)
    payload["finished_at_unix"] = time.time()
    payload["receipt_self_sha256"] = _self_hash(payload)
    report_path = output / "campaign.json"
    if report_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {report_path}; pass --force")
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("runs/gpu-campaign"))
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-uncached", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--matrix", type=Path, default=Path("configs/data/realistic-agent-public-eval-matrix.v1.json"))
    parser.add_argument("--transfer", action="append", help="BASE:TARGET checkpoint pair; repeat")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--acquire-hf", action="store_true", help="download approved HF sources")
    parser.add_argument("--hf-config", type=Path, default=Path("configs/experiments/hf-sources.v1.yaml"))
    parser.add_argument("--hf-out", type=Path, default=Path("data/hf-campaign"))
    parser.add_argument("--hf-source", action="append", dest="hf_sources")
    parser.add_argument("--hf-dry-run", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="log campaign scalars to W&B")
    parser.add_argument("--wandb-project", default="localagent")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_campaign(args)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"output": str(Path(args.output) / "campaign.json"), "sha256": report["receipt_self_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
