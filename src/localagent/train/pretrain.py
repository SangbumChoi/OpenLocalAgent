"""Pretrain from scratch: next-token CE over packed, document-split shards.

The compact in-memory stream API remains useful for tests. Production/config-driven runs use
``PackedShardDataset``: memory-mapped rows, masked padding, deterministic sampling, gradient
accumulation, held-out validation, and resumable checkpoints.
"""

from __future__ import annotations

import random
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from localagent.train.device import autocast_ctx
from localagent.train.loop import cosine_lr, router_loss_terms, set_lr, wsd_lr


def _verify_configured_corpus_freeze(
    data_config: Mapping[str, Any],
    *,
    project_root: str | Path = ".",
) -> dict[str, Any] | None:
    """Fail closed on an explicitly configured content-addressed corpus freeze."""

    configured = data_config.get("corpus_freeze")
    if configured is None:
        return None
    if not isinstance(configured, Mapping):
        raise ValueError("data.corpus_freeze must be a mapping")
    freeze_path = configured.get("path")
    spec_path = configured.get("spec")
    if not isinstance(freeze_path, str) or not freeze_path:
        raise ValueError("data.corpus_freeze.path must be a non-empty path string")
    if not isinstance(spec_path, str) or not spec_path:
        raise ValueError("data.corpus_freeze.spec must be a non-empty path string")

    from localagent.data.corpus_freeze import verify_corpus_freeze

    freeze = verify_corpus_freeze(
        freeze_path,
        spec_path,
        project_root=project_root,
    )
    return {
        "path": freeze_path,
        "spec": spec_path,
        "sha256": freeze["freeze_sha256"],
    }


def _full_lm_token_counts(x: torch.Tensor, y: torch.Tensor) -> tuple[int, int]:
    """Count real inputs and next-token targets in a dense or right-padded full-LM batch."""

    supervised_per_row = (y != -100).sum(dim=1)
    # A partially filled packed row has one final real input with no next-token target. Full rows
    # already have one target per input, hence the clamp at the tensor width.
    input_per_row = torch.clamp(supervised_per_row + 1, max=x.shape[1])
    return int(input_per_row.sum()), int(supervised_per_row.sum())


def _validate_exact_resume_checkpoint(
    checkpoint: Any,
    *,
    model,
    seed: int,
    use_grad_scaler: bool,
    device: torch.device,
) -> Mapping[str, Any]:
    """Reject partial checkpoints rather than silently degrading exact resume."""

    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "exact pretraining resume currently supports CPU, CUDA, and MPS only; "
            f"{device.type!r} accelerator RNG restoration is not implemented"
        )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("exact-resume checkpoint root must be a mapping")
    required = {
        "cfg",
        "cuda_rng_state_all",
        "eval_rng_state",
        "grad_scaler",
        "loss_history",
        "optimizer",
        "rng_state",
        "stage",
        "state_dict",
        "step",
        "token_accounting",
        "torch_rng_state",
        "training_seed",
        "validation_history",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            "exact-resume checkpoint is incomplete; missing: " + ", ".join(missing)
        )
    if checkpoint.get("stage") != "pretrain":
        raise ValueError("exact-resume checkpoint stage must be 'pretrain'")
    if checkpoint.get("cfg") != model.cfg.__dict__:
        raise ValueError("exact-resume checkpoint model config mismatch")
    if not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError("exact-resume checkpoint state_dict is invalid")
    if not isinstance(checkpoint.get("optimizer"), Mapping):
        raise ValueError("exact-resume checkpoint optimizer state is invalid")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("exact-resume checkpoint step is invalid")
    if checkpoint.get("training_seed") != seed:
        raise ValueError(
            "resume checkpoint training seed mismatch: "
            f"recorded={checkpoint.get('training_seed')!r}, expected={seed!r}"
        )
    loss_history = checkpoint.get("loss_history")
    if not isinstance(loss_history, list) or len(loss_history) != step + 1:
        raise ValueError("exact-resume checkpoint loss history is inconsistent")
    if not isinstance(checkpoint.get("validation_history"), list):
        raise ValueError("exact-resume checkpoint validation history is invalid")
    accounting = checkpoint.get("token_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("exact-resume checkpoint token accounting is invalid")
    for key in ("input_tokens", "loss_tokens"):
        value = accounting.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"exact-resume checkpoint token accounting {key!r} is invalid"
            )
    if checkpoint.get("rng_state") is None or checkpoint.get("eval_rng_state") is None:
        raise ValueError("exact-resume checkpoint Python RNG state is missing")
    torch_rng_state = checkpoint.get("torch_rng_state")
    if (
        not isinstance(torch_rng_state, torch.Tensor)
        or torch_rng_state.dtype != torch.uint8
        or torch_rng_state.ndim != 1
    ):
        raise ValueError("exact-resume checkpoint Torch RNG state is invalid")
    scaler_state = checkpoint.get("grad_scaler")
    if use_grad_scaler and not isinstance(scaler_state, Mapping):
        raise ValueError("exact-resume checkpoint gradient-scaler state is missing")
    if not use_grad_scaler and scaler_state is not None:
        raise ValueError("exact-resume checkpoint has unexpected gradient-scaler state")
    cuda_rng_state = checkpoint.get("cuda_rng_state_all")
    if device.type == "cuda" and not isinstance(cuda_rng_state, (list, tuple)):
        raise ValueError("exact-resume checkpoint CUDA RNG state is missing")
    if device.type != "cuda" and cuda_rng_state is not None:
        raise ValueError("exact-resume checkpoint has unexpected CUDA RNG state")
    mps_rng_state = checkpoint.get("mps_rng_state")
    if device.type == "mps":
        if (
            not isinstance(mps_rng_state, torch.Tensor)
            or mps_rng_state.dtype != torch.uint8
            or mps_rng_state.ndim != 1
        ):
            raise ValueError("exact-resume checkpoint MPS RNG state is missing or invalid")
    elif mps_rng_state is not None:
        raise ValueError("exact-resume checkpoint has unexpected MPS RNG state")
    return checkpoint


def _paths_alias(first: Path, second: Path) -> bool:
    """Return whether two configured artifact paths resolve to the same file."""

    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


def pretrain(model, stream, tok, *, steps=400, batch_size=32, seq_len=128, lr=3e-3,
             warmup=30, device="cpu", log=print, lr_schedule="cosine", decay_frac=0.2,
             accum_steps=1, weight_decay=0.1, grad_clip=1.0, seed=0,
             val_data=None, eval_every=0, eval_batches=8, checkpoint_path=None,
             checkpoint_every=0, resume_from=None, amp_dtype=torch.float32,
             checkpoint_mirror_path=None, lineage=None, tokenizer_metadata=None,
             data_metadata=None, execution=None, return_metrics=False):
    """Next-token CE over a packed byte stream. `lr_schedule="wsd"` (opt-in, MiniCPM) replaces the
    cosine LR with Warmup-Stable-Decay (warmup -> flat plateau -> exponential `lr*0.5^((s-S)/T)`
    over the last `decay_frac` of steps); default "cosine" is the legacy schedule.

    ``stream`` can be a one-dimensional token sequence or any object implementing
    ``sample_batch(batch_size, rng, device)`` (notably ``PackedShardDataset``).
    ``steps`` counts optimizer updates, not micro-batches.
    """
    if lr_schedule not in ("cosine", "wsd"):
        raise ValueError(f"pretrain() lr_schedule must be 'cosine' or 'wsd', got {lr_schedule!r}")
    if accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")
    torch.manual_seed(seed)
    model.train()
    model.to(device)
    device_obj = torch.device(device)
    use_grad_scaler = device_obj.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    packed = hasattr(stream, "sample_batch")
    data = None if packed else torch.tensor(stream, dtype=torch.long)
    n = 0 if data is None else data.numel()
    if not packed:
        assert n > seq_len + 1, "pretrain stream too short"
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 1)
    hist = []
    lm_loss_history: list[float] = []
    router_aux_loss_history: list[float] = []
    router_weighted_loss_history: list[float] = []
    validation_history: list[dict[str, Any]] = []
    start_step = 0
    input_tokens_seen = 0
    loss_tokens_seen = 0
    if resume_from is not None:
        from localagent.train.stage_data import assert_resume_lineage

        checkpoint = _validate_exact_resume_checkpoint(
            torch.load(resume_from, map_location="cpu", weights_only=True),
            model=model,
            seed=seed,
            use_grad_scaler=use_grad_scaler,
            device=device_obj,
        )
        recorded_lineage = checkpoint.get("lineage")
        if lineage is not None:
            assert_resume_lineage(checkpoint, lineage)
        elif recorded_lineage is not None:
            raise ValueError(
                "resume checkpoint records lineage but no expected lineage was provided"
            )
        recorded_execution = checkpoint.get("execution")
        if execution is not None:
            if not isinstance(recorded_execution, Mapping):
                raise ValueError(
                    "resume checkpoint has no execution identity for an exact continuation"
                )
            if dict(recorded_execution) != dict(execution):
                raise ValueError("resume checkpoint execution identity mismatch")
        elif recorded_execution is not None:
            raise ValueError(
                "resume checkpoint records execution identity but none was provided"
            )
        model.load_state_dict(checkpoint["state_dict"])
        opt.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        saved_accounting = checkpoint.get("token_accounting")
        input_tokens_seen = int(saved_accounting["input_tokens"])
        loss_tokens_seen = int(saved_accounting["loss_tokens"])
        hist = list(checkpoint["loss_history"])
        loss_components = checkpoint.get("loss_components")
        if loss_components is None:
            if model.cfg.sparse_ffn:
                raise ValueError(
                    "sparse pretrain resume checkpoint is missing router loss components"
                )
            lm_loss_history = list(hist)
            router_aux_loss_history = [0.0] * len(hist)
            router_weighted_loss_history = [0.0] * len(hist)
        else:
            if not isinstance(loss_components, Mapping):
                raise ValueError("pretrain resume loss_components must be a mapping")
            component_names = {
                "lm_loss": lm_loss_history,
                "router_aux": router_aux_loss_history,
                "router_weighted": router_weighted_loss_history,
            }
            for name, destination in component_names.items():
                values = loss_components.get(name)
                if not isinstance(values, list) or len(values) != len(hist):
                    raise ValueError(
                        f"pretrain resume loss_components[{name!r}] is inconsistent"
                    )
                destination.extend(float(value) for value in values)
        validation_history = [
            {
                "step": int(record["step"]),
                "loss": float(record["loss"]),
                "batches": int(record.get("batches", eval_batches)),
            }
            for record in checkpoint["validation_history"]
        ]
        rng.setstate(checkpoint["rng_state"])
        eval_rng.setstate(checkpoint["eval_rng_state"])
        if use_grad_scaler:
            scaler.load_state_dict(checkpoint["grad_scaler"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device_obj.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        elif device_obj.type == "mps":
            torch.mps.set_rng_state(checkpoint["mps_rng_state"].cpu())
        if start_step > steps:
            raise ValueError(
                f"resume checkpoint is already at step {start_step - 1}, beyond total steps {steps}"
            )

    def batch_from(source, sampler_rng=rng):
        if hasattr(source, "sample_batch"):
            return source.sample_batch(batch_size, sampler_rng, device)
        source_tensor = data if source is stream else torch.tensor(source, dtype=torch.long)
        source_n = source_tensor.numel()
        starts = [
            sampler_rng.randint(0, source_n - seq_len - 2) for _ in range(batch_size)
        ]
        batch = torch.stack(
            [source_tensor[start:start + seq_len + 1] for start in starts]
        ).to(device)
        return batch[:, :-1], batch[:, 1:]

    @torch.no_grad()
    def evaluate(source) -> float:
        was_training = model.training
        model.eval()
        losses = []
        for _ in range(eval_batches):
            x_val, y_val = batch_from(source, eval_rng)
            with autocast_ctx(device_obj, amp_dtype):
                _, val_loss = model(x_val, targets=y_val)
            losses.append(float(val_loss.detach()))
        model.train(was_training)
        return sum(losses) / len(losses)

    def save(step: int) -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if device_obj.type == "mps":
            # Materialize all queued optimizer writes before taking one coherent checkpoint.
            torch.mps.synchronize()
        payload = {
            "cfg": model.cfg.__dict__,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            # ``tokens_seen`` remains the legacy alias for optimizer-supervised tokens.
            "tokens_seen": loss_tokens_seen,
            "input_tokens_seen": input_tokens_seen,
            "token_accounting": {
                "input_tokens": input_tokens_seen,
                "loss_tokens": loss_tokens_seen,
                "sources": {
                    "train": {
                        "input_tokens": input_tokens_seen,
                        "loss_tokens": loss_tokens_seen,
                    }
                },
            },
            "loss_history": hist,
            "loss_components": {
                "lm_loss": lm_loss_history,
                "router_aux": router_aux_loss_history,
                "router_weighted": router_weighted_loss_history,
            },
            "validation_history": validation_history,
            "rng_state": rng.getstate(),
            "eval_rng_state": eval_rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device_obj.type == "cuda" else None
            ),
            "mps_rng_state": (
                torch.mps.get_rng_state() if device_obj.type == "mps" else None
            ),
            "grad_scaler": scaler.state_dict() if use_grad_scaler else None,
            "stage": "pretrain",
            "training_seed": seed,
            "lineage": lineage,
        }
        if tokenizer_metadata is not None:
            payload["tokenizer"] = dict(tokenizer_metadata)
        if data_metadata is not None:
            payload["data"] = dict(data_metadata)
        if execution is not None:
            payload["execution"] = dict(execution)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        if checkpoint_mirror_path is not None:
            mirror = Path(checkpoint_mirror_path)
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror_tmp = mirror.with_suffix(mirror.suffix + ".tmp")
            shutil.copy2(path, mirror_tmp)
            mirror_tmp.replace(mirror)

    for step in range(start_step, steps):
        if lr_schedule == "wsd":
            set_lr(opt, wsd_lr(step, steps, lr, warmup, decay_frac, min_ratio=0.0))
        else:
            set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_lm_loss = 0.0
        step_router_aux = 0.0
        step_router_weighted = 0.0
        for _ in range(accum_steps):
            x, y = batch_from(stream)
            with autocast_ctx(device_obj, amp_dtype):
                _, lm_loss = model(x, targets=y)
                loss, router_aux, router_weighted = router_loss_terms(model, lm_loss)
            scaler.scale(loss / accum_steps).backward()
            step_loss += float(loss.detach()) / accum_steps
            step_lm_loss += float(lm_loss.detach()) / accum_steps
            step_router_aux += float(router_aux.detach()) / accum_steps
            step_router_weighted += float(router_weighted.detach()) / accum_steps
            input_count, loss_count = _full_lm_token_counts(x, y)
            input_tokens_seen += input_count
            loss_tokens_seen += loss_count
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        hist.append(step_loss)
        lm_loss_history.append(step_lm_loss)
        router_aux_loss_history.append(step_router_aux)
        router_weighted_loss_history.append(step_router_weighted)
        should_eval = bool(val_data is not None and eval_every and (
            step % eval_every == 0 or step == steps - 1
        ))
        val_loss = None
        if should_eval:
            val_loss = evaluate(val_data)
            validation_history.append(
                {"step": step, "loss": val_loss, "batches": eval_batches}
            )
        should_log = step % max(1, steps // 8) == 0 or step == steps - 1 or should_eval
        if should_log:
            message = (
                f"  [pretrain] step {step:4d}/{steps}  loss {step_loss:.3f}  "
                f"loss_tokens {loss_tokens_seen:,}"
            )
            if model.cfg.sparse_ffn:
                message += (
                    f"  lm {step_lm_loss:.3f}  router_aux {step_router_aux:.3f}  "
                    f"router_weighted {step_router_weighted:.3f}"
                )
            if val_loss is not None:
                message += f"  val {val_loss:.3f}"
            log(message)
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            save(step)
    if steps > 0:
        save(steps - 1)
    metrics: dict[str, Any] = {
        "training_seed": seed,
        "steps_completed": len(hist),
        "token_accounting": {
            "input_tokens": input_tokens_seen,
            "loss_tokens": loss_tokens_seen,
            "sources": {
                "train": {
                    "input_tokens": input_tokens_seen,
                    "loss_tokens": loss_tokens_seen,
                }
            },
        },
        "validation_history": validation_history,
        "validation_last": validation_history[-1] if validation_history else None,
        "loss_components": {
            "optimization": hist,
            "lm_loss": lm_loss_history,
            "router_aux": router_aux_loss_history,
            "router_weighted": router_weighted_loss_history,
        },
    }
    return (hist, metrics) if return_metrics else hist


def run(config_path: str, *, resume: bool | None = None) -> None:
    """Run pretraining from YAML.

    ``runtime.resume`` restores the exact local checkpoint, including optimizer/RNG state.
    Top-level ``init_from`` instead validates and loads only compatible parent weights before a
    fresh optimizer is constructed.
    """

    import json

    import yaml

    from localagent.data.pretrain_corpus import PackedShardDataset
    from localagent.model import LocalAgentLM, ModelConfig
    from localagent.model.tokenizer import load_tokenizer
    from localagent.train.checkpoint_growth import (
        checkpoint_tokenizer_sha256,
        load_checkpoint_with_sha256,
        verify_growth_checkpoint,
    )
    from localagent.train.device import execution_metadata, resolve_device, resolve_dtype
    from localagent.train.midtrain import assert_checkpoint_compatible
    from localagent.train.stage_data import (
        build_stage_lineage,
        canonical_sha256,
        tokenizer_identity,
    )

    config = yaml.safe_load(Path(config_path).read_text())
    cfg = ModelConfig.from_yaml(config["model_config"])
    cfg.assert_within_budget()
    data_cfg = config["data"]
    corpus_freeze = _verify_configured_corpus_freeze(data_cfg)
    train_data = PackedShardDataset(data_cfg["shards_dir"], "train")
    available_train_tokens = int(train_data.manifest["splits"]["train"]["tokens"])
    minimum_train_tokens = data_cfg.get("min_train_tokens")
    if minimum_train_tokens is not None:
        if (
            isinstance(minimum_train_tokens, bool)
            or not isinstance(minimum_train_tokens, int)
            or minimum_train_tokens <= 0
        ):
            raise ValueError("data.min_train_tokens must be a positive integer")
        if available_train_tokens < minimum_train_tokens:
            raise ValueError(
                "packed training corpus is smaller than data.min_train_tokens: "
                f"available={available_train_tokens:,}, "
                f"required={minimum_train_tokens:,}"
            )
    val_data = None
    if train_data.manifest["splits"].get("val", {}).get("rows", 0):
        val_data = PackedShardDataset(data_cfg["shards_dir"], "val")
    if train_data.seq_len > cfg.max_seq_len:
        raise ValueError(
            f"packed seq_len {train_data.seq_len} exceeds model max_seq_len {cfg.max_seq_len}"
        )
    if int(train_data.manifest["vocab_size"]) != cfg.vocab_size:
        raise ValueError("packed corpus vocabulary does not match model config")
    tok_cfg = data_cfg.get("tokenizer", {"kind": "byte"})
    tok = load_tokenizer(tok_cfg.get("kind", "byte"), tok_cfg.get("path"))
    if tok.vocab_size != cfg.vocab_size:
        raise ValueError("tokenizer vocabulary does not match model config")
    tokenizer_lineage = tokenizer_identity(
        str(tok_cfg.get("kind", "byte")),
        vocab_size=tok.vocab_size,
        path=tok_cfg.get("path"),
    )

    runtime = config.get("runtime", {})
    requested_device = runtime.get("device", "auto")
    requested_dtype = runtime.get("dtype", "auto")
    device = resolve_device(requested_device)
    dtype = resolve_dtype(device, requested_dtype)
    execution = execution_metadata(
        requested_device=requested_device,
        resolved_device=device,
        requested_dtype=requested_dtype,
        resolved_dtype=dtype,
    )
    seed = int(runtime.get("seed", 0))
    torch.manual_seed(seed)
    model = LocalAgentLM(cfg)
    schedule = config.get("schedule", {})
    batch = config.get("batch", {})
    optim = config.get("optim", {})
    log_cfg = config.get("log", {})
    out_dir = Path(log_cfg.get("out_dir", "runs/pretrain"))
    checkpoint = out_dir / "latest.pt"
    mirror_dir = log_cfg.get("mirror_dir")
    mirror_checkpoint = Path(mirror_dir) / "latest.pt" if mirror_dir else None
    init_value = config.get("init_from")
    if init_value is not None and (not isinstance(init_value, str) or not init_value):
        raise ValueError("pretrain init_from must be a non-empty checkpoint path string")
    init_from = Path(init_value) if init_value is not None else None
    if init_from is not None and _paths_alias(init_from, checkpoint):
        raise ValueError("pretrain init_from must differ from its output checkpoint; use resume")
    if (
        init_from is not None
        and mirror_checkpoint is not None
        and _paths_alias(init_from, mirror_checkpoint)
    ):
        raise ValueError("pretrain init_from must differ from its mirror checkpoint")
    if mirror_checkpoint is not None and _paths_alias(mirror_checkpoint, checkpoint):
        raise ValueError("pretrain output and mirror checkpoints must be different")
    configured_resume = runtime.get("resume", False)
    if not isinstance(configured_resume, bool):
        raise TypeError("runtime.resume must be boolean")
    resume_requested = configured_resume if resume is None else resume
    if not isinstance(resume_requested, bool):
        raise TypeError("resume override must be boolean or None")
    if resume is True and not checkpoint.exists():
        raise FileNotFoundError(
            f"pretrain resume requested but checkpoint does not exist: {checkpoint}"
        )
    resume_from = checkpoint if resume_requested and checkpoint.exists() else None
    loaded_parent_sha256 = None
    if init_from is not None and resume_from is None:
        parent, loaded_parent_sha256 = load_checkpoint_with_sha256(init_from)
        if parent.get("growth") is not None or parent.get("stage") == "checkpoint_growth":
            verify_growth_checkpoint(parent)
        assert_checkpoint_compatible(parent, cfg)
        parent_tokenizer_sha256 = checkpoint_tokenizer_sha256(parent)
        if parent_tokenizer_sha256 != tokenizer_lineage["sha256"]:
            raise ValueError(
                "init_from checkpoint tokenizer lineage does not match configured tokenizer"
            )
        state = parent.get("state_dict", parent.get("model"))
        if not isinstance(state, Mapping):
            raise ValueError("pretrain init_from checkpoint has no state_dict/model mapping")
        model.load_state_dict(state)
    manifest_sha256 = canonical_sha256(train_data.manifest)
    lineage = build_stage_lineage(
        stage="pretrain",
        config=config,
        model_config=cfg.__dict__,
        data_identity={
            "kind": "packed_shards",
            "manifest_sha256": manifest_sha256,
            "split": train_data.split,
            "corpus_freeze": corpus_freeze,
        },
        tokenizer=tokenizer_lineage,
        workspace=Path(__file__).resolve(),
        parent_checkpoint=init_from,
    )
    if (
        loaded_parent_sha256 is not None
        and lineage.get("parent_checkpoint_sha256") != loaded_parent_sha256
    ):
        raise ValueError("pretrain init_from checkpoint changed while it was being validated")
    loss_history, metrics = pretrain(
        model,
        train_data,
        tok,
        steps=int(schedule.get("total_steps", 20_000)),
        batch_size=int(batch.get("micro_batch_size", 8)),
        seq_len=train_data.seq_len,
        lr=float(optim.get("lr", 3e-4)),
        warmup=int(schedule.get("warmup_steps", 200)),
        device=device,
        lr_schedule=str(schedule.get("type", "cosine")),
        decay_frac=float(schedule.get("decay_frac", 0.2)),
        accum_steps=int(batch.get("grad_accum_steps", 1)),
        weight_decay=float(optim.get("weight_decay", 0.1)),
        grad_clip=float(optim.get("grad_clip", 1.0)),
        seed=seed,
        val_data=val_data,
        eval_every=int(log_cfg.get("eval_every", 0)),
        eval_batches=int(log_cfg.get("eval_batches", 8)),
        checkpoint_path=checkpoint,
        checkpoint_every=int(log_cfg.get("ckpt_every", 0)),
        resume_from=resume_from,
        amp_dtype=dtype,
        checkpoint_mirror_path=mirror_checkpoint,
        lineage=lineage,
        tokenizer_metadata={
            "kind": str(tok_cfg.get("kind", "byte")),
            "path": tok_cfg.get("path"),
            "sha256": tokenizer_lineage["sha256"],
        },
        data_metadata={
            "kind": "packed_shards",
            "path": str(data_cfg["shards_dir"]),
            "split": train_data.split,
            "manifest_sha256": manifest_sha256,
            "available_train_tokens": available_train_tokens,
            "minimum_train_tokens": minimum_train_tokens,
            "corpus_freeze": corpus_freeze,
        },
        execution=execution,
        return_metrics=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "stage": "pretrain",
                "checkpoint": str(checkpoint),
                "loss_last": loss_history[-1] if loss_history else None,
                "loss_steps": len(loss_history),
                **metrics,
                "lineage": lineage,
                "execution": execution,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
