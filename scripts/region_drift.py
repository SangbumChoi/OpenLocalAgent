#!/usr/bin/env python
"""How far each donor region moves during SFT, and how much of the model it is.

A region is worth loading when the fine-tune keeps what the donor put there. This measures, per
region, the relative L2 displacement and cosine similarity between the public donor and a trained
arm, alongside the region's parameter share — the two quantities that explain the transfer ranking.

  python scripts/region_drift.py --checkpoint runs/region/data-union/model.pt --out runs/region/drift.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

DONOR = Path("data/hf-campaign/localagent_tiny_model/model.safetensors")

REGIONS = {
    "embedding": lambda key: key.startswith("embed"),
    "attention": lambda key: ".attn." in key,
    "ffn": lambda key: ".ffn." in key,
    "norms": lambda key: "norm" in key,
}


def layer_index(key: str) -> int | None:
    parts = key.split(".")
    return int(parts[1]) if parts[0] == "blocks" and parts[1].isdigit() else None


def summarize(donor: dict, trained: dict, keys: list[str]) -> dict[str, float]:
    donor_flat = torch.cat([donor[k].float().flatten() for k in keys])
    trained_flat = torch.cat([trained[k].float().flatten() for k in keys])
    delta = trained_flat - donor_flat
    return {
        "parameters": int(donor_flat.numel()),
        "donor_l2": float(donor_flat.norm()),
        "delta_l2": float(delta.norm()),
        "relative_delta_l2": float(delta.norm() / max(donor_flat.norm(), 1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(
            donor_flat.unsqueeze(0), trained_flat.unsqueeze(0)).item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    donor = load_file(str(DONOR))
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    shared = [k for k in donor if k in trained and donor[k].shape == trained[k].shape]
    total = sum(donor[k].numel() for k in shared)

    report = {"checkpoint": args.checkpoint, "shared_tensors": len(shared), "parameters": total,
              "by_region": {}, "by_layer": {}}
    for name, rule in REGIONS.items():
        keys = [k for k in shared if rule(k)]
        if keys:
            row = summarize(donor, trained, keys)
            row["parameter_share"] = row["parameters"] / total
            report["by_region"][name] = row
    layers = sorted({index for k in shared if (index := layer_index(k)) is not None})
    for index in layers:
        keys = [k for k in shared if layer_index(k) == index]
        row = summarize(donor, trained, keys)
        row["parameter_share"] = row["parameters"] / total
        report["by_layer"][str(index)] = row

    for name, row in report["by_region"].items():
        print(f"{name:10s} share={row['parameter_share']*100:5.1f}%  "
              f"rel_delta={row['relative_delta_l2']:.4f}  cos={row['cosine']:.4f}", flush=True)
    print("layer drift:", " ".join(
        f"{i}:{report['by_layer'][i]['relative_delta_l2']:.3f}" for i in report["by_layer"]),
        flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("DRIFT_DONE " + args.out, flush=True)


if __name__ == "__main__":
    main()
