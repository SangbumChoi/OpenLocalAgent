#!/usr/bin/env python3
"""Summarize current warm/random transfer evidence into an adoption decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CURRENT_SHA = "d6b5df5ff0ba53cb797501b21fd788ca904b2773cf7ea56269864d0350670e3c"
RANDOM_SHA = "390f1414260e118cd621af735fe6e87b01e8641b1cff650d594585e39b212e45"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _groups(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    groups = report["warm"]["weight_transfer"]["groups"]
    return {
        name: {
            "parameters": int(item["parameters"]),
            "relative_delta_l2": float(item["relative_delta_l2"]),
            "delta_l2": float(item["delta_l2"]),
        }
        for name, item in groups.items()
    }


def assemble(
    android_path: Path,
    mcp_path: Path,
    enterprise_path: Path,
    agentnet_path: Path,
    mobilegym_path: Path,
    browsergym_path: Path,
    output: Path,
) -> dict[str, Any]:
    android = _load(android_path)
    mcp = _load(mcp_path)
    enterprise = _load(enterprise_path)
    agentnet = _load(agentnet_path)
    mobilegym = _load(mobilegym_path)
    browsergym = _load(browsergym_path)
    if android["warm"]["child"]["sha256"] != CURRENT_SHA:
        raise ValueError("AndroidControl warm child is not current m626")
    if android["random"]["child"]["sha256"] != RANDOM_SHA:
        raise ValueError("AndroidControl random child mismatch")
    if mcp["parent_checkpoint"]["sha256"] != CURRENT_SHA:
        raise ValueError("MCP continuation is not based on current m626")
    if enterprise["warm_checkpoint"]["identity"]["sha256"] != CURRENT_SHA:
        raise ValueError("EnterpriseOps warm checkpoint mismatch")
    if enterprise["random_checkpoint"]["identity"]["sha256"] != RANDOM_SHA:
        raise ValueError("EnterpriseOps random checkpoint mismatch")
    if agentnet["warm_full"]["checkpoint"]["sha256"] != CURRENT_SHA:
        raise ValueError("AgentNet warm checkpoint mismatch")
    if mobilegym["checkpoint"]["sha256"] != CURRENT_SHA or browsergym["checkpoint"]["sha256"] != CURRENT_SHA:
        raise ValueError("native deployment receipts are not current")
    body_groups = {"androidcontrol": _groups(android), "mcpmark": _groups(mcp)}
    body_names = ["embedding", "attention_or_mixer", "ffn", "normalization"]
    mean_body_movement = {
        name: sum(body_groups[surface][name]["relative_delta_l2"] for surface in body_groups) / len(body_groups)
        for name in body_names
    }
    payload: dict[str, Any] = {
        "kind": "localagent_m641_current_weight_adoption_summary",
        "schema_version": 1,
        "current_checkpoint": {"sha256": CURRENT_SHA, "parameters": 10_524_544},
        "public_transfer_evidence": {
            "androidcontrol": {
                "dataset_url": android["source"]["url"],
                "original_url": android["source"]["original_url"],
                "warm_minus_random_after_pp": android["comparison"]["aggregate"]["warm_minus_random_after_pp"],
                "warm_eval_token_accuracy": android["comparison"]["aggregate"]["warm_after_token_accuracy"],
                "random_eval_token_accuracy": android["comparison"]["aggregate"]["random_after_token_accuracy"],
            },
            "mcp_trajectory": {
                "dataset_url": mcp["source"]["url"],
                "warm_minus_random_after_pp": mcp["comparison"]["aggregate"]["warm_minus_random_after_pp"],
                "warm_eval_token_accuracy": mcp["comparison"]["aggregate"]["warm_after_token_accuracy"],
                "random_eval_token_accuracy": mcp["comparison"]["aggregate"]["random_after_token_accuracy"],
            },
            "enterpriseopsgym_email": {
                "dataset_url": enterprise["benchmark"]["dataset_url"],
                "warm_minus_random_top1_pp": enterprise["warm_minus_random_delta"]["hit_at_1"] * 100,
                "warm_top1": enterprise["warm_checkpoint"]["summary"]["hit_at_1"],
                "random_top1": enterprise["random_checkpoint"]["summary"]["hit_at_1"],
            },
            "agentnet_text_projection": {
                "dataset_url": agentnet["benchmark"]["dataset_url"],
                "matched_parent_count": agentnet["matched_subset"]["parents"],
                "warm_minus_random_first_action_type": agentnet["matched_subset"]["warm_minus_random"]["first_action_type_rate"],
                "warm_minus_random_mean_total": agentnet["matched_subset"]["warm_minus_random"]["mean_total"],
                "full_warm_exact_trajectory_rate": agentnet["warm_full"]["overall"]["exact_trajectory_rate"],
            },
        },
        "shared_body_movement_relative_l2": {
            "by_surface": body_groups,
            "mean_android_mcp": mean_body_movement,
            "action_heads": "0.0 in both transfer reports; action heads were held fixed",
        },
        "native_guardrails": {
            "mobilegym": {
                "success_rate": mobilegym["success_rate"],
                "tasks": mobilegym["task_count"],
                "state_grounding_transfer": mobilegym["diagnosis"]["state_grounding_transfer"],
            },
            "browsergym_miniwob": {
                "success_rate": browsergym["success_rate"],
                "episodes": browsergym["task_count"],
                "grounded_steps": browsergym["result"]["grounded_steps"],
                "noop_or_ungrounded_steps": browsergym["result"]["noop_or_ungrounded_steps"],
            },
        },
        "adoption_decision": {
            "reuse_current_warm_backbone": True,
            "freeze_or_low_lr_shared_body": True,
            "initialize_new_heads_from_controlled_seed": True,
            "use_larger_head_lr_than_transferred_body": True,
            "export_to_webgpu_as_native_agent": False,
            "reason": (
                "Warm initialization wins on four source-linked text/tool projections, while shared-body "
                "movement stays small and action heads remain fixed. Native MobileGym and BrowserGym "
                "success remain low, so this supports initialization policy—not native deployment readiness."
            ),
        },
        "inputs": {
            label: _identity(path)
            for label, path in (
                ("androidcontrol", android_path),
                ("mcpmark", mcp_path),
                ("enterpriseopsgym", enterprise_path),
                ("agentnet", agentnet_path),
                ("mobilegym", mobilegym_path),
                ("browsergym", browsergym_path),
            )
        },
        "claim_boundary": (
            "Cross-dataset weight-adoption analysis bound to the current m626 checkpoint. It combines "
            "source-linked projection ablations with native MobileGym/BrowserGym guardrails; it does "
            "not combine unlike metrics into a leaderboard score and does not establish AndroidWorld, "
            "AgentNetBench, OSWorld, MCPMark, EnterpriseOps-Gym server, or real-account success."
        ),
    }
    payload["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("android", "mcp", "enterprise", "agentnet", "mobilegym", "browsergym"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.android, args.mcp, args.enterprise, args.agentnet, args.mobilegym, args.browsergym, args.out)
    print(json.dumps(report["adoption_decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
