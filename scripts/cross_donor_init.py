#!/usr/bin/env python
"""Initialise our architecture from a *different* family's open-source checkpoint.

The premise of the report is that a practitioner designs a small architecture for their own
latency budget and then wants the open weights that already exist. Those weights never match the
new shape, so something has to project them. We slice rather than rotate: a truncated SVD would
choose a different basis for every matrix independently and destroy the alignment between
consecutive layers, whereas taking the leading sub-block keeps every layer in one coordinate
subspace and then rescales to the student's initialisation variance.

Donor layers are mapped onto student layers at even spacing, and the embedding is transferred by
matching token *strings* across the two tokenizers, which is the only correspondence that survives
a vocabulary change.

  python scripts/cross_donor_init.py --donor data/baselines/Qwen2.5-0.5B-Instruct \
      --student runs/mix-10m-hybrid-seed2026/latest.pt --region blocks \
      --out runs/donor-init/qwen25-05b-blocks/latest.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localagent.model import LocalAgentLM, ModelConfig

# Student tensor role -> the donor tensors that play the same role, in concat order.
ROLE_MAP = {
    "attn.in_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "attn.out_proj": ("self_attn.o_proj",),
    "ffn.gate": ("mlp.gate_proj",),
    "ffn.up": ("mlp.up_proj",),
    "ffn.down": ("mlp.down_proj",),
}
REGIONS = {
    "attn": ("attn.in_proj", "attn.out_proj"),
    "ffn": ("ffn.gate", "ffn.up", "ffn.down"),
    "blocks": tuple(ROLE_MAP),
}
# Which depth band of the student receives donor weights; the rest keeps its random
# initialisation. Answering "which layers carry transferable information" needs the bands varied
# one at a time, because transferring everything cannot attribute the result to a depth.
BANDS = ("all", "early", "middle", "late")


def band_layers(band: str, n_layers: int) -> set[int]:
    if band == "all":
        return set(range(n_layers))
    third = max(1, round(n_layers / 3))
    spans = {"early": range(0, third),
             "middle": range(third, min(n_layers, 2 * third)),
             "late": range(min(n_layers, 2 * third), n_layers)}
    return set(spans[band])


def fit(donor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The leading sub-block of the donor, rescaled to the target's standard deviation."""
    sliced = donor
    for axis, width in enumerate(target.shape):
        sliced = sliced.narrow(axis, 0, min(width, sliced.shape[axis]))
    out = target.clone()
    region = [slice(0, size) for size in sliced.shape]
    scale = target.std() / sliced.std().clamp_min(1e-8)
    out[tuple(region)] = (sliced * scale).to(target.dtype)
    return out


def donor_tensors(path: Path) -> dict[str, torch.Tensor]:
    """Donor weights from either a Hugging Face directory or one of our own checkpoints.

    A same-family donor is the cleanest way to vary the donor/student size ratio without also
    changing the architecture, so a `.pt` checkpoint is accepted alongside safetensors.
    """
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload["state_dict"]
    from safetensors.torch import load_file

    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors in {path}")
    weights: dict[str, torch.Tensor] = {}
    for shard in shards:
        weights.update(load_file(str(shard)))
    return weights


def donor_layer_count(weights: dict) -> int:
    indices = {int(key.split(".")[2]) for key in weights
               if key.startswith("model.layers.") and key.count(".") > 3}
    indices |= {int(key.split(".")[1]) for key in weights
                if key.startswith("blocks.") and key.count(".") > 2}
    return max(indices) + 1 if indices else 0


def same_family_key(role: str, layer: int) -> str:
    """Our own checkpoints name the roles directly, so no translation table is needed."""
    return {"attn.in_proj": f"blocks.{layer}.attn.in_proj.weight",
            "attn.out_proj": f"blocks.{layer}.attn.out_proj.weight",
            "ffn.gate": f"blocks.{layer}.ffn.gate.weight",
            "ffn.up": f"blocks.{layer}.ffn.up.weight",
            "ffn.down": f"blocks.{layer}.ffn.down.weight"}[role]


def transfer_embedding(weights: dict, state: dict, donor_path: Path, tokenizer_path: str) -> int:
    """Copy donor embedding rows for tokens whose surface string exists in both vocabularies."""
    from transformers import AutoTokenizer

    donor_embed = weights.get("model.embed_tokens.weight")
    if donor_embed is None or "embed.weight" not in state:
        return 0
    donor_vocab = AutoTokenizer.from_pretrained(str(donor_path)).get_vocab()
    # The student's tokenizer object exposes no vocab mapping, but its on-disk file is the
    # standard tokenizers format, which does.
    student_vocab = json.loads(Path(tokenizer_path).read_text())["model"]["vocab"]

    target = state["embed.weight"]
    width = min(target.shape[1], donor_embed.shape[1])
    scale = target.std() / donor_embed.std().clamp_min(1e-8)
    copied = 0
    for token, student_id in student_vocab.items():
        donor_id = donor_vocab.get(token)
        if donor_id is None or student_id >= target.shape[0]:
            continue
        target[student_id, :width] = (donor_embed[donor_id, :width] * scale).to(target.dtype)
        copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True, help="a Hugging Face checkpoint directory")
    ap.add_argument("--student", required=True, help="a checkpoint whose cfg defines the student")
    ap.add_argument("--region", default="blocks", choices=sorted(REGIONS) + ["embed", "all"])
    ap.add_argument("--band", default="all", choices=BANDS,
                    help="which depth third of the student receives donor weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer-h100-16k.json")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    checkpoint = torch.load(args.student, map_location="cpu", weights_only=False)
    torch.manual_seed(args.seed)
    fields = {key: value for key, value in checkpoint["cfg"].items()
              if key in ModelConfig.__dataclass_fields__}
    student = LocalAgentLM(ModelConfig(**fields))
    state = {key: tensor.clone() for key, tensor in student.state_dict().items()}

    donor_path = Path(args.donor)
    native = donor_path.suffix == ".pt"
    weights = donor_tensors(donor_path)
    donor_layers = donor_layer_count(weights)
    student_layers = student.cfg.n_layers
    if donor_layers == 0:
        raise SystemExit("donor exposes no model.layers.* tensors")

    roles = REGIONS.get(args.region, tuple(ROLE_MAP))
    copied, params = [], 0
    if args.region in ("embed", "all") and native:
        # Same family, same tokenizer: the embedding transfers by slicing, no string matching.
        donor_embed = weights.get("embed.weight")
        if donor_embed is not None and "embed.weight" in state:
            state["embed.weight"] = fit(donor_embed.float(), state["embed.weight"])
            copied.append("embed.weight")
            params += state["embed.weight"].numel()
    elif args.region in ("embed", "all"):
        rows = transfer_embedding(weights, state, donor_path, args.tokenizer)
        copied.append(f"embed:{rows}rows")
        params += rows * state["embed.weight"].shape[1]
    receiving = band_layers(args.band, student_layers)
    if args.region != "embed":
        for student_index in sorted(receiving):
            # Even spacing: a 4-layer student takes the donor's first, middle and last thirds.
            donor_index = min(donor_layers - 1,
                              round(student_index * (donor_layers - 1) / max(1, student_layers - 1)))
            for role in roles:
                key = f"blocks.{student_index}.{role}.weight"
                if key not in state:
                    continue
                if native:
                    sources = [weights.get(same_family_key(role, donor_index))]
                else:
                    sources = [weights.get(f"model.layers.{donor_index}.{name}.weight")
                               for name in ROLE_MAP[role]]
                sources = [tensor for tensor in sources if tensor is not None]
                if not sources:
                    continue
                merged = torch.cat(sources, dim=0) if len(sources) > 1 else sources[0]
                state[key] = fit(merged.float(), state[key])
                copied.append(key)
                params += state[key].numel()

    checkpoint["state_dict"] = state
    checkpoint["optimizer"] = None
    checkpoint["donor_transfer"] = {
        "donor": str(donor_path), "donor_is_same_family": native,
        "region": args.region, "band": args.band,
        "layers_receiving": sorted(receiving), "donor_layers": donor_layers,
        "student_layers": student_layers, "tensors_copied": len(copied),
        "params_touched": params,
        "params_total": sum(tensor.numel() for tensor in state.values()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    print(json.dumps(checkpoint["donor_transfer"], indent=2))
    print("DONOR_INIT_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
