"""Fail-closed workshop/publication readiness checks for realistic-agent evidence.

The realistic-agent catalog and the individual result bridges deliberately allow local protocol
receipts without pretending that they are native benchmark runs.  This module is the final join:
it requires explicit native-environment receipts, a real WebGPU capability/performance receipt, a
transfer-vs-no-transfer ablation, a successful current-checkpoint RL preflight, and a public
artifact manifest before reporting ``ready``.

An absent receipt is a blocked requirement, never an implicit pass.  The gate is intentionally
independent of any benchmark package and does not execute an emulator, browser, VM, MCP server, or
Hugging Face upload.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from localagent.eval.realistic_preflight import preflight_catalog

SCHEMA_VERSION = 1
KIND = "localagent_workshop_publication_gate"

# These are the smallest set of native gates that covers the user's requested deployment surface:
# mobile UI, browser/desktop control, stateful tools, and realistic email/productivity workflows.
REQUIRED_NATIVE_BENCHMARKS: tuple[str, ...] = (
    "androidworld",
    "mobilegym",
    "mobile_safety_bench",
    "iosworld",
    "browsergym_miniwob",
    "osworld",
    "osworld_v2",
    "agentnet",
    "toolsandbox",
    "mcpmark",
    "enterpriseopsgym",
)

_SHA256_HEX = frozenset("0123456789abcdef")
_NATIVE_MISSING = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file() or path.is_symlink():
        return None, "missing_or_non_regular_file"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(value, Mapping):
        return None, "json_root_must_be_object"
    return dict(value), None


def _check(
    requirement: str,
    *,
    status: str,
    evidence: Sequence[str] = (),
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": sorted(set(str(item) for item in evidence)),
        "blockers": sorted(set(str(item) for item in blockers)),
    }


def _receipt_checkpoint_sha256(payload: Mapping[str, Any]) -> str | None:
    """Return the evaluated checkpoint hash from common native-receipt layouts."""

    direct = payload.get("checkpoint_sha256")
    if isinstance(direct, str):
        return direct
    direct = payload.get("current_checkpoint_sha256")
    if isinstance(direct, str):
        return direct
    for key in ("checkpoint", "model"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            for field in ("sha256", "checkpoint_sha256", "current_checkpoint_sha256"):
                value = nested.get(field)
                if isinstance(value, str):
                    return value
    return None


def _native_field(payload: Mapping[str, Any], field: str) -> Any:
    """Read a native receipt field from legacy flat or canonical nested layouts."""

    if field in payload:
        return payload[field]
    for section in ("environment", "result", "summary"):
        nested = payload.get(section)
        if isinstance(nested, Mapping) and field in nested:
            return nested[field]
    return _NATIVE_MISSING


def _native_receipt_check(
    benchmark_id: str,
    path: Path | None,
    *,
    repo_root: Path,
    current_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    requirement = f"native:{benchmark_id}"
    if path is None:
        return _check(requirement, status="blocked", blockers=["receipt_not_supplied"])
    resolved = path if path.is_absolute() else repo_root / path
    payload, error = _read_json(resolved)
    if payload is None:
        return _check(
            requirement,
            status="blocked",
            evidence=[str(resolved)],
            blockers=[error or "receipt_unreadable"],
        )
    required = {
        "benchmark_id",
        "environment_executed",
        "official_split_verified",
        "task_count",
        "success_rate",
    }
    missing = sorted(field for field in required if _native_field(payload, field) is _NATIVE_MISSING)
    if missing:
        return _check(
            requirement,
            status="blocked",
            evidence=[str(resolved), _sha256(resolved)],
            blockers=[f"missing_field:{field}" for field in missing],
        )
    blockers: list[str] = []
    if _native_field(payload, "benchmark_id") != benchmark_id:
        blockers.append("benchmark_id_mismatch")
    if _native_field(payload, "environment_executed") is not True:
        blockers.append("environment_not_executed")
    if _native_field(payload, "official_split_verified") is not True:
        blockers.append("official_split_not_verified")
    task_count = _native_field(payload, "task_count")
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 1:
        blockers.append("task_count_not_positive")
    success_rate = _native_field(payload, "success_rate")
    if (
        isinstance(success_rate, bool)
        or not isinstance(success_rate, (int, float))
        or not 0.0 <= float(success_rate) <= 1.0
    ):
        blockers.append("success_rate_not_in_0_1")
    if current_checkpoint_sha256 is not None:
        receipt_checkpoint = _receipt_checkpoint_sha256(payload)
        if receipt_checkpoint is None:
            blockers.append("current_checkpoint_not_bound")
        elif receipt_checkpoint != current_checkpoint_sha256:
            blockers.append("current_checkpoint_sha256_mismatch")
    return _check(
        requirement,
        status="pass" if not blockers else "blocked",
        evidence=[str(resolved), _sha256(resolved)],
        blockers=blockers,
    )


def _webgpu_check(path: Path | None, *, repo_root: Path) -> dict[str, Any]:
    requirement = "webgpu:native_capability_and_latency"
    if path is None:
        return _check(requirement, status="blocked", blockers=["receipt_not_supplied"])
    resolved = path if path.is_absolute() else repo_root / path
    payload, error = _read_json(resolved)
    if payload is None:
        return _check(
            requirement,
            status="blocked",
            evidence=[str(resolved)],
            blockers=[error or "receipt_unreadable"],
        )
    blockers: list[str] = []
    if payload.get("backend") != "webgpu":
        blockers.append("backend_not_explicit_webgpu")
    if payload.get("environment_executed") is not True:
        blockers.append("environment_not_executed")
    if payload.get("hardware_adapter") in (None, "", "software", "swiftshader"):
        blockers.append("hardware_adapter_not_verified")
    capability = payload.get("capability")
    if not isinstance(capability, Mapping):
        blockers.append("missing_capability_metrics")
    else:
        for field in ("evaluated_cases", "exact_actions", "closed_loop_success"):
            value = capability.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                blockers.append(f"invalid_capability:{field}")
    performance = payload.get("performance")
    if not isinstance(performance, Mapping):
        blockers.append("missing_performance_metrics")
    else:
        if not isinstance(performance.get("tokens_per_second_p50"), (int, float)):
            blockers.append("missing_performance:tokens_per_second_p50")
        if not isinstance(performance.get("peak_memory_mb"), (int, float)):
            blockers.append("missing_performance:peak_memory_mb")
    return _check(
        requirement,
        status="pass" if not blockers else "blocked",
        evidence=[str(resolved), _sha256(resolved)],
        blockers=blockers,
    )


def _weight_check(
    paths: Sequence[Path],
    *,
    repo_root: Path,
    current_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    requirement = "weights:transfer_and_no_transfer_ablation"
    if not paths:
        return _check(requirement, status="blocked", blockers=["two_reports_required"])
    reports: list[tuple[Path, dict[str, Any]]] = []
    unified_reports: list[tuple[Path, dict[str, Any]]] = []
    blockers: list[str] = []
    for path in paths:
        resolved = path if path.is_absolute() else repo_root / path
        payload, error = _read_json(resolved)
        if payload is None:
            blockers.append(f"unreadable:{resolved}:{error}")
            continue
        unified = payload.get("weight_transfer_analysis")
        comparison = payload.get("comparison")
        if isinstance(unified, Mapping) and isinstance(comparison, Mapping):
            # The current cross-surface receipt nests the warm/random compatibility reports
            # rather than using the legacy top-level ``ablation``/``held_out`` fields.
            for arm in ("warm", "random"):
                arm_report = unified.get(arm)
                if not isinstance(arm_report, Mapping):
                    blockers.append(f"missing_unified_arm:{resolved}:{arm}")
                    continue
                compatibility = arm_report.get("compatibility")
                if not isinstance(compatibility, Mapping):
                    blockers.append(f"missing_unified_compatibility:{resolved}:{arm}")
                    continue
                if compatibility.get("config_mismatches") not in ({}, None):
                    blockers.append(f"config_mismatch:{resolved}:{arm}")
                if compatibility.get("shape_mismatches") not in ({}, None):
                    blockers.append(f"shape_mismatch:{resolved}:{arm}")
                if compatibility.get("tokenizer_sha256_equal") is not True:
                    blockers.append(f"tokenizer_mismatch:{resolved}:{arm}")
                groups = arm_report.get("groups")
                if not isinstance(groups, Mapping) or not groups:
                    blockers.append(f"missing_unified_weight_groups:{resolved}:{arm}")
            contract = comparison.get("arm_contract")
            if not isinstance(contract, Mapping):
                blockers.append(f"missing_unified_arm_contract:{resolved}")
            else:
                if contract.get("warm_backbone_init") != "parent":
                    blockers.append(f"invalid_unified_warm_init:{resolved}")
                if contract.get("random_backbone_init") != "random":
                    blockers.append(f"invalid_unified_random_init:{resolved}")
            parent = payload.get("parent_checkpoint")
            if current_checkpoint_sha256 is not None:
                parent_sha256 = parent.get("sha256") if isinstance(parent, Mapping) else None
                if parent_sha256 is None:
                    blockers.append(f"unified_current_checkpoint_not_bound:{resolved}")
                elif parent_sha256 != current_checkpoint_sha256:
                    blockers.append(f"unified_current_checkpoint_sha256_mismatch:{resolved}")
            unified_reports.append((resolved, payload))
            continue
        reports.append((resolved, payload))
    if unified_reports and not reports:
        return _check(
            requirement,
            status="pass" if not blockers else "blocked",
            evidence=[f"{path}:{_sha256(path)}" for path, _ in unified_reports],
            blockers=blockers,
        )
    # A canonical combined receipt may carry both parent-head and random/no-transfer arms.  This
    # avoids making reviewers reconstruct an ablation from several opaque files while retaining
    # the same compatibility checks as the per-transition analyzer.
    if len(reports) == 1:
        resolved, payload = reports[0]
        compatibility = payload.get("compatibility")
        held_out = payload.get("held_out")
        ablation = payload.get("ablation")
        if not isinstance(compatibility, Mapping):
            blockers.append(f"missing_compatibility:{resolved}")
        else:
            if compatibility.get("config_mismatches") not in ({}, None):
                blockers.append(f"config_mismatch:{resolved}")
            if compatibility.get("shape_mismatches") not in ({}, None):
                blockers.append(f"shape_mismatch:{resolved}")
            if compatibility.get("tokenizer_sha256_equal") is not True:
                blockers.append(f"tokenizer_mismatch:{resolved}")
        if not isinstance(ablation, Mapping):
            blockers.append(f"missing_combined_ablation:{resolved}")
        if not isinstance(held_out, Mapping) or not {"parent_heads", "random"}.issubset(held_out):
            blockers.append(f"combined_ablation_arms_missing:{resolved}")
        else:
            for arm in ("parent_heads", "random"):
                if not isinstance(held_out[arm], Mapping):
                    blockers.append(f"invalid_ablation_arm:{arm}:{resolved}")
        return _check(
            requirement,
            status="pass" if not blockers else "blocked",
            evidence=[f"{path}:{_sha256(path)}" for path, _ in reports],
            blockers=blockers,
        )

    if len(reports) < 2:
        blockers.append("two_reports_required")
    for resolved, payload in reports:
        compatibility = payload.get("compatibility")
        if not isinstance(compatibility, Mapping):
            blockers.append(f"missing_compatibility:{resolved}")
            continue
        if compatibility.get("config_mismatches") not in ({}, None):
            blockers.append(f"config_mismatch:{resolved}")
        if compatibility.get("shape_mismatches") not in ({}, None):
            blockers.append(f"shape_mismatch:{resolved}")
        if compatibility.get("tokenizer_sha256_equal") is not True:
            blockers.append(f"tokenizer_mismatch:{resolved}")
        if not isinstance(payload.get("ablation"), str) and not isinstance(
            payload.get("probe_init"), str
        ):
            blockers.append(f"missing_ablation_label:{resolved}")
        if not isinstance(payload.get("held_out"), Mapping) and not isinstance(
            payload.get("heldout"), Mapping
        ):
            blockers.append(f"missing_held_out_metrics:{resolved}")
    return _check(
        requirement,
        status="pass" if not blockers and (len(reports) >= 2 or unified_reports) else "blocked",
        evidence=[
            *[f"{path}:{_sha256(path)}" for path, _ in reports],
            *[f"{path}:{_sha256(path)}" for path, _ in unified_reports],
        ],
        blockers=(
            blockers
            if blockers or len(reports) >= 2 or unified_reports
            else ["two_reports_required"]
        ),
    )


def _public_artifact_check(
    path: Path | None,
    *,
    repo_root: Path,
    current_checkpoint_sha256: str | None = None,
    current_checkpoint_error: str | None = None,
) -> dict[str, Any]:
    requirement = "artifacts:public_model_demo_manifest"
    if path is None:
        return _check(requirement, status="blocked", blockers=["manifest_not_supplied"])
    resolved = path if path.is_absolute() else repo_root / path
    payload, error = _read_json(resolved)
    if payload is None:
        return _check(
            requirement,
            status="blocked",
            evidence=[str(resolved)],
            blockers=[error or "manifest_unreadable"],
        )
    required = {"public", "model_url", "demo_url", "artifact_sha256"}
    blockers = [f"missing_field:{field}" for field in sorted(required - set(payload))]
    if payload.get("public") is not True:
        blockers.append("public_flag_not_true")
    digest = payload.get("artifact_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _SHA256_HEX for character in digest)
    ):
        blockers.append("artifact_sha256_invalid")
    for field in ("model_url", "demo_url"):
        if not isinstance(payload.get(field), str) or not payload[field].startswith(("https://", "http://")):
            blockers.append(f"{field}_invalid")
    if current_checkpoint_error is not None:
        blockers.append(current_checkpoint_error)
    elif current_checkpoint_sha256 is not None:
        published_checkpoint = payload.get("current_checkpoint_sha256")
        if published_checkpoint is None:
            blockers.append("current_checkpoint_not_bound")
        elif published_checkpoint != current_checkpoint_sha256:
            blockers.append("current_checkpoint_sha256_mismatch")
    return _check(
        requirement,
        status="pass" if not blockers else "blocked",
        evidence=[str(resolved), _sha256(resolved)],
        blockers=blockers,
    )


def _rl_preflight_parent_sha256(payload: Mapping[str, Any]) -> str | None:
    """Return the checkpoint used as the RL preflight parent, if recorded."""

    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        lineage = metrics.get("lineage")
        if isinstance(lineage, Mapping):
            value = lineage.get("parent_checkpoint_sha256")
            if isinstance(value, str):
                return value
    source = payload.get("source")
    if isinstance(source, Mapping):
        parent = source.get("sft_parent_checkpoint")
        if isinstance(parent, Mapping):
            value = parent.get("sha256")
            if isinstance(value, str):
                return value
    return None


def _rl_preflight_check(
    path: Path | None,
    *,
    repo_root: Path,
    current_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Require a successful, current-checkpoint-bound one-update RL preflight."""

    requirement = "training:rl_preflight"
    if path is None:
        return _check(requirement, status="blocked", blockers=["receipt_not_supplied"])
    resolved = path if path.is_absolute() else repo_root / path
    payload, error = _read_json(resolved)
    if payload is None:
        return _check(
            requirement,
            status="blocked",
            evidence=[str(resolved)],
            blockers=[error or "receipt_unreadable"],
        )
    blockers: list[str] = []
    if payload.get("kind") != "localagent_one_update_training_preflight":
        blockers.append("preflight_kind_mismatch")
    if payload.get("status") != "passed":
        blockers.append("preflight_status_not_passed")
    if current_checkpoint_sha256 is not None:
        parent_sha256 = _rl_preflight_parent_sha256(payload)
        if parent_sha256 is None:
            blockers.append("current_checkpoint_not_bound")
        elif parent_sha256 != current_checkpoint_sha256:
            blockers.append("current_checkpoint_sha256_mismatch")
    return _check(
        requirement,
        status="pass" if not blockers else "blocked",
        evidence=[str(resolved), _sha256(resolved)],
        blockers=blockers,
    )


def build_workshop_gate(
    catalog_path: str | Path = "configs/data/realistic-agent-eval.catalog.yaml",
    *,
    repo_root: str | Path = ".",
    native_receipts: Mapping[str, str | Path] | None = None,
    webgpu_receipt: str | Path | None = None,
    weight_reports: Sequence[str | Path] = (),
    public_artifact_manifest: str | Path | None = None,
    rl_preflight_receipt: str | Path | None = None,
    current_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-compatible, fail-closed workshop readiness report."""

    root = Path(repo_root).resolve()
    catalog_source = Path(catalog_path)
    if not catalog_source.is_absolute():
        catalog_source = root / catalog_source
    catalog = preflight_catalog(catalog_source)
    checks: list[dict[str, Any]] = []
    required_families = {"mobile", "browser", "computer", "tool_api"}
    observed_families = {
        row["family"] for row in catalog["rows"] if isinstance(row.get("family"), str)
    }
    missing_families = sorted(required_families - observed_families)
    checks.append(
        _check(
            "catalog:realistic_family_coverage",
            status="pass" if not missing_families else "blocked",
            evidence=[str(catalog_path), catalog["catalog_sha256"]],
            blockers=[f"missing_family:{family}" for family in missing_families],
        )
    )
    checks.append(
        _check(
            "catalog:no_pending_train_adapter",
            status="pass" if catalog["counts"]["runnable"] >= catalog["counts"]["train_rows"] else "blocked",
            evidence=[str(catalog_path)],
            blockers=[
                f"train_row_blocked:{row['id']}"
                for row in catalog["rows"]
                if row["train_policy"] == "train" and not row["runnable"]
            ],
        )
    )
    current_checkpoint_sha256: str | None = None
    current_checkpoint_error: str | None = None
    current_checkpoint_path: Path | None = None
    if current_checkpoint is not None:
        checkpoint_path = Path(current_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        current_checkpoint_path = checkpoint_path
        if checkpoint_path.is_file() and not checkpoint_path.is_symlink():
            current_checkpoint_sha256 = _sha256(checkpoint_path)
        else:
            current_checkpoint_error = "current_checkpoint_unreadable"
    receipts = dict(native_receipts or {})
    for benchmark_id in REQUIRED_NATIVE_BENCHMARKS:
        checks.append(
            _native_receipt_check(
                benchmark_id,
                Path(receipts[benchmark_id]) if benchmark_id in receipts else None,
                repo_root=root,
                current_checkpoint_sha256=current_checkpoint_sha256,
            )
        )
    checks.append(
        _webgpu_check(
            Path(webgpu_receipt) if webgpu_receipt is not None else None,
            repo_root=root,
        )
    )
    checks.append(
        _weight_check(
            [Path(path) for path in weight_reports],
            repo_root=root,
            current_checkpoint_sha256=current_checkpoint_sha256,
        )
    )
    checks.append(
        _rl_preflight_check(
            Path(rl_preflight_receipt) if rl_preflight_receipt is not None else None,
            repo_root=root,
            current_checkpoint_sha256=current_checkpoint_sha256,
        )
    )
    checks.append(
        _public_artifact_check(
            Path(public_artifact_manifest)
            if public_artifact_manifest is not None
            else None,
            repo_root=root,
            current_checkpoint_sha256=current_checkpoint_sha256,
            current_checkpoint_error=current_checkpoint_error,
        )
    )
    blocking = [
        {
            "requirement": check["requirement"],
            "blockers": check["blockers"],
        }
        for check in checks
        if check["status"] != "pass"
    ]
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "ready": not blocking,
        "catalog": {
            "path": str(Path(catalog_path)),
            "sha256": catalog["catalog_sha256"],
            "entries": catalog["catalog_entries"],
            "runnable_ids": catalog["runnable_ids"],
            "blocked_ids": catalog["blocked_ids"],
        },
        "current_checkpoint": (
            {
                "path": str(current_checkpoint_path),
                "sha256": current_checkpoint_sha256,
            }
            if current_checkpoint_path is not None
            else None
        ),
        "checks": checks,
        "blocking_requirements": blocking,
        "claim_boundary": (
            "ready is false unless native benchmark receipts, native WebGPU performance and "
            "capability evidence, transfer ablations, a successful current-checkpoint RL "
            "preflight, and public model/demo artifacts are all "
            "explicitly supplied; protocol bridges and synthetic receipts do not satisfy these checks"
        ),
    }


def write_workshop_gate(report: Mapping[str, Any], path: str | Path) -> None:
    """Write a stable workshop gate report without silently overwriting an existing file."""

    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite workshop gate output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
