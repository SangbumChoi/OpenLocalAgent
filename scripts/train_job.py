"""Standalone single-process pipeline for a REMOTE runner (HF Jobs, Colab).

This is the path for a machine you do not control: one process, one config, no queue.
On the training box use scripts/train_queue.sh instead - it resumes, locks per job,
chains the stages and scores them.

Its --stages vocabulary (pretrain,sft,grpo) predates openlocalagent.stages.Stage and is
kept because the launchers and any saved job command line pass it verbatim.
"""
import argparse
import os
import time

import torch

from openlocalagent.data.render import build_pretrain_stream
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import load_tokenizer
from openlocalagent.train.pretrain import pretrain
from openlocalagent.train.rl import grpo
from openlocalagent.train.sft import sft

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/model/tiny-30m-byte.yaml")
ap.add_argument("--stages", default="pretrain,sft,grpo")
ap.add_argument("--real", action="store_true", help="use real HF datasets (needs `datasets`)")
ap.add_argument("--quick", action="store_true", help="tiny CPU smoke")
ap.add_argument("--out", default="results/runs/job")
ap.add_argument("--push", default="", help="HF model repo to push the final checkpoint to")
ap.add_argument("--init-hub", default="", help="resume: 'repo:file.pt' to load + skip pretrain")
# stage budgets (full-run defaults; --quick shrinks them)
ap.add_argument("--pretrain-steps", type=int, default=6000)
ap.add_argument("--sft-steps", type=int, default=3000)
ap.add_argument("--grpo-steps", type=int, default=300)
ap.add_argument("--seq-len", type=int, default=512)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--no-amp", action="store_true", help="disable mixed precision (force fp32)")
ap.add_argument("--dtype", default="auto", help="auto|bf16|fp16|fp32 autocast dtype")
ap.add_argument("--compile", action="store_true", help="torch.compile the pretrain/SFT forward")
args = ap.parse_args()

from openlocalagent.train.device import enable_tf32  # noqa: E402

enable_tf32()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = not args.no_amp and DEVICE != "cpu"                 # AMP only helps on accelerators
if args.quick:
    args.pretrain_steps, args.sft_steps, args.grpo_steps = 30, 30, 6
    args.batch, args.seq_len = 16, 128
stages = set(args.stages.split(","))
os.makedirs(args.out, exist_ok=True)
tok = load_tokenizer()
cfg = ModelConfig.from_yaml(args.config)
model = LocalAgentLM(cfg).to(DEVICE)
if args.init_hub:                                         # resume from a pushed checkpoint
    from huggingface_hub import hf_hub_download
    repo, fname = args.init_hub.split(":")
    ick = torch.load(hf_hub_download(repo, fname), map_location=DEVICE)
    model.load_state_dict(ick["state_dict"])
    stages.discard("pretrain")                            # already pretrained
    print(f"resumed from {args.init_hub}; skipping pretrain", flush=True)
# torch.compile wraps only the dense pretrain/SFT forward; `model` (raw) still owns the weights and
# is used for save + GRPO rollouts (KV-cache decoding recompiles badly under compile).
fwd = torch.compile(model) if args.compile else model
print(f"device={DEVICE} cfg={cfg.name} params~{cfg.estimate_params()/1e6:.1f}M stages={sorted(stages)} "
      f"real={args.real} quick={args.quick} amp={AMP} compile={args.compile}", flush=True)


_api = None


def save(tag):
    p = f"{args.out}/{cfg.name}-{tag}.pt"
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, p)
    print(f"  saved {p}", flush=True)
    if args.push:                       # push each stage immediately so progress survives a later crash
        global _api
        try:
            from huggingface_hub import HfApi
            _api = _api or HfApi()
            _api.create_repo(args.push, repo_type="model", exist_ok=True)
            _api.upload_file(path_or_fileobj=p, path_in_repo=f"{tag}.pt", repo_id=args.push)
            print(f"  pushed {tag}.pt -> {args.push}", flush=True)
        except Exception as e:          # noqa: BLE001 - a push hiccup must not kill training
            print(f"  (push of {tag} failed: {e})", flush=True)
    return p


def synth_samples(n):
    from openlocalagent.data.synth.agent_synth import Generator
    return Generator(level=3, seed=7).generate_balanced(n)


def real_sft_data(n):
    """Real public function-calling SFT: try Hermes (public) then xLAM (gated); skip on failure."""
    from openlocalagent.data.corpus.hf_datasets import hermes_sft_samples, xlam_sft_samples
    for loader in (hermes_sft_samples, xlam_sft_samples):
        try:
            rows = loader(tok, n=n)
            if rows:
                return rows
        except Exception as e:          # noqa: BLE001 - gated/unavailable/schema -> next option
            print(f"  (SFT loader {loader.__name__} unavailable: {e})", flush=True)
    return None


t0 = time.time()
# ---- 1. PRETRAIN (raw text -> next-byte LM) ----
if "pretrain" in stages:
    print("\n=== PRETRAIN ===", flush=True)
    if args.real:
        from openlocalagent.data.corpus.hf_datasets import fineweb_byte_stream
        stream = fineweb_byte_stream(tok, max_chars=2_000_000 if args.quick else 200_000_000)
    else:
        stream = build_pretrain_stream(synth_samples(2 if args.quick else 40), tok)
    hold = min(len(stream) // 10, 500_000)                 # held-out tail for BPB
    val, train_stream = stream[-hold:], stream[:-hold] if hold else stream
    pretrain(fwd, train_stream, tok, steps=args.pretrain_steps, batch_size=args.batch,
             seq_len=args.seq_len, device=DEVICE, lr_schedule="wsd", amp=AMP, amp_dtype=args.dtype)
    if val:
        from openlocalagent.eval.bpb import bits_per_byte
        print(f"  held-out BPB = {bits_per_byte(model, val, seq_len=args.seq_len, device=DEVICE):.3f} "
              f"bits/byte", flush=True)
    save("pretrain")

# ---- 2. SFT (tool-call instruction tuning) ----
if "sft" in stages:
    print("\n=== SFT ===", flush=True)
    from openlocalagent.data.render import render_sft
    sft_data = real_sft_data(500 if args.quick else 60000) if args.real else []
    # drop samples whose rendered length overflows the model context (real fn-calling prompts can be
    # very long: full in-context tool schemas), then mix in synthetic so SFT always has enough.
    sft_data = [s for s in (sft_data or []) if len(render_sft(s, tok)[0]) <= cfg.max_seq_len]
    if len(sft_data) < (5 if args.quick else 2000):
        print(f"  ({len(sft_data)} real SFT rows fit ctx -> mixing in synthetic)", flush=True)
        sft_data += synth_samples(2 if args.quick else 40)
    sft(fwd, sft_data, tok, steps=args.sft_steps, batch_size=max(8, args.batch // 2),
        device=DEVICE, amp=AMP, amp_dtype=args.dtype)
    save("sft")

# ---- 3. GRPO (RL with a verifiable tool-call reward) ----
if "grpo" in stages:
    print("\n=== GRPO (verifiable reward) ===", flush=True)
    rl_data = [s for s in synth_samples(1 if args.quick else 20) if s.kind == "tool"]
    grpo(model, rl_data, tok, steps=args.grpo_steps, device=DEVICE,
         prompts_per_step=4 if args.quick else 8, amp=AMP, amp_dtype=args.dtype)
    save("grpo")

final = save("final")
print(f"\nALL STAGES DONE in {(time.time()-t0)/60:.1f} min -> {final}", flush=True)

# ---- optional: push to the Hub ----
if args.push:
    import json

    from safetensors.torch import save_file
    json.dump(cfg.__dict__, open(f"{args.out}/config.json", "w"), indent=2)
    save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()},
              f"{args.out}/model.safetensors")
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.push, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=final, path_in_repo="model.pt", repo_id=args.push)
    api.upload_file(path_or_fileobj=f"{args.out}/config.json", path_in_repo="config.json", repo_id=args.push)
    api.upload_file(path_or_fileobj=f"{args.out}/model.safetensors", path_in_repo="model.safetensors", repo_id=args.push)
    print(f"PUSHED -> https://huggingface.co/{args.push}", flush=True)
print("JOB_DONE", flush=True)
