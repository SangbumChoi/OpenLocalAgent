"""LocalAgent CLI. `localagent <command> ...`

Commands:
  model-info <model.yaml>      Construct a config and report param count vs the 100M budget.
  train <stage> <config.yaml>  Run a pipeline stage (pretrain|midtrain|sft|distill|rl|eval).
  synth <config.yaml>          Generate synthetic agent data.
  eval <checkpoint>            Run the eval harness.
  export <target> <ckpt> <out> Export to gguf|onnx|executorch.
  grow-checkpoint <src> ...    Warm-start a deeper compatible model with an explicit layer map.
  chat <checkpoint>            Launch the terminal agent demo.
"""

from __future__ import annotations

import argparse

from localagent import __version__


def _model_info(args) -> None:
    from localagent.model.config import ModelConfig

    cfg = ModelConfig.from_yaml(args.config)
    n = cfg.estimate_params()
    cfg.assert_within_budget()
    print(f"{cfg.name}: ~{n/1e6:.2f}M params (budget 100M) — OK")
    if cfg.sparse_ffn:
        print(
            f"  active/token≈{cfg.estimate_active_params()/1e6:.2f}M params "
            f"(router + top-{cfg.ffn_top_k}/{cfg.ffn_num_experts} experts per layer)"
        )
    print(f"  d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads}/{cfg.n_kv_heads} "
          f"ffn={cfg.ffn_hidden} vocab={cfg.vocab_size}")
    extras = []
    if cfg.factorized:
        extras.append(f"factorized embed_dim={cfg.embed_dim}")
    if cfg.n_loops > 1:
        extras.append(f"depth-recurrence x{cfg.n_loops} (effective depth {cfg.effective_depth})")
    if cfg.sparse_ffn:
        extras.append(
            f"sparse FFN top-{cfg.ffn_top_k}/{cfg.ffn_num_experts} "
            f"router_aux={cfg.router_aux_loss_coef:g}"
        )
    if extras:
        print("  " + " · ".join(extras))
    print(
        f"  fp16 weights={cfg.estimate_weight_bytes(16)/1e6:.1f}MB "
        f"q4 weights≈{cfg.estimate_weight_bytes(4)/1e6:.1f}MB "
        f"fp16 cache@{cfg.max_seq_len}={cfg.estimate_cache_bytes(cfg.max_seq_len)/1e6:.1f}MB"
    )


def _train(args) -> None:
    from localagent.pipeline.flow import run_stage

    resume_git_receipt = getattr(args, "resume_git_receipt", None)
    if resume_git_receipt is None:
        run_stage(args.stage, args.config, resume=args.resume)
    else:
        run_stage(
            args.stage,
            args.config,
            resume=args.resume,
            resume_git_receipt=resume_git_receipt,
        )


def _create_resume_git_receipt(args) -> None:
    from localagent.train.sft import create_resume_git_receipt

    create_resume_git_receipt(
        args.config,
        args.out,
        reason=args.reason,
        evidence=args.evidence,
    )


def _synth(args) -> None:
    from localagent.data.agent_synth import synthesize

    synthesize(args.config)


def _eval(args) -> None:
    from localagent.eval.harness import run

    run(args.checkpoint)


def _export(args) -> None:
    import importlib

    mod = importlib.import_module(f"localagent.inference.export.to_{args.target}")
    mod.export(args.checkpoint, args.out)


def _grow_checkpoint(args) -> None:
    import json

    from localagent.model.config import ModelConfig
    from localagent.train.checkpoint_growth import write_grown_checkpoint

    cfg = ModelConfig.from_yaml(args.target_config)
    cfg.assert_within_budget()
    manifest = write_grown_checkpoint(
        args.source_checkpoint,
        cfg,
        args.layer_map,
        args.out,
        overwrite=args.force,
    )
    print(
        json.dumps(
            {"checkpoint": args.out, "growth": manifest},
            indent=2,
            sort_keys=True,
        )
    )


def _parse_layer_map_arg(value: str) -> dict[int, int]:
    from localagent.train.checkpoint_growth import parse_layer_map

    try:
        return parse_layer_map(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _chat(args) -> None:
    raise NotImplementedError("TODO(phase-7): load ckpt + Agent + REPL (see demos/chat_cli.py)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="localagent", description=__doc__)
    p.add_argument("--version", action="version", version=f"localagent {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    mi = sub.add_parser("model-info", help="report param count vs budget")
    mi.add_argument("config")
    mi.set_defaults(func=_model_info)

    tr = sub.add_parser("train", help="run a pipeline stage")
    tr.add_argument("stage", choices=["pretrain", "midtrain", "sft", "distill", "rl", "eval"])
    tr.add_argument("config")
    tr.add_argument(
        "--resume",
        action="store_true",
        help="require and exactly resume the stage's existing latest.pt checkpoint",
    )
    tr.add_argument(
        "--resume-git-receipt",
        metavar="PATH",
        help="authorize one SFT resume across receipt-proven Git worktree-only drift",
    )
    tr.set_defaults(func=_train)

    rr = sub.add_parser(
        "create-resume-git-receipt",
        help="create a non-overwriting external receipt for SFT Git-only resume drift",
    )
    rr.add_argument("config")
    rr.add_argument("out")
    rr.add_argument("--reason", required=True)
    rr.add_argument(
        "--evidence",
        action="append",
        required=True,
        help="repeatable evidence statement supporting the non-numerical migration",
    )
    rr.set_defaults(func=_create_resume_git_receipt)

    sy = sub.add_parser("synth", help="generate synthetic agent data")
    sy.add_argument("config")
    sy.set_defaults(func=_synth)

    ev = sub.add_parser("eval", help="run eval harness")
    ev.add_argument("checkpoint")
    ev.set_defaults(func=_eval)

    ex = sub.add_parser("export", help="export to an on-device runtime")
    ex.add_argument("target", choices=["gguf", "onnx", "executorch", "hf"])
    ex.add_argument("checkpoint")
    ex.add_argument("out")
    ex.set_defaults(func=_export)

    gr = sub.add_parser(
        "grow-checkpoint",
        help="warm-start a deeper compatible model (not function-preserving)",
    )
    gr.add_argument("source_checkpoint")
    gr.add_argument("target_config")
    gr.add_argument("out")
    gr.add_argument(
        "--layer-map",
        required=True,
        type=_parse_layer_map_arg,
        metavar="TARGET:SOURCE,...",
        help="explicit mapping for every target layer, e.g. 0:0,1:0,2:1",
    )
    gr.add_argument("--force", action="store_true", help="replace an existing output checkpoint")
    gr.set_defaults(func=_grow_checkpoint)

    ch = sub.add_parser("chat", help="terminal agent demo")
    ch.add_argument("checkpoint")
    ch.set_defaults(func=_chat)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
