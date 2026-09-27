#!/usr/bin/env python
"""Export a trained LocalAgent checkpoint to a Hugging Face Hub model repo.

Builds the bundle (config + weights + heads + model card) and, with --push + a token, uploads it.

  # local bundle only (no token needed):
  python scripts/push_to_hf.py --checkpoint runs/flywheel/ultra-tiny.pt --out runs/hf_export

  # push (needs HF_TOKEN env, --token, or `hf auth login`):
  python scripts/push_to_hf.py --checkpoint runs/flywheel/ultra-tiny.pt \
      --repo <user>/localagent-ultra-tiny-1m --push
"""

from __future__ import annotations

import argparse
import os

from localagent.inference.export.to_hf import export_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/flywheel/ultra-tiny.pt")
    ap.add_argument("--out", default="runs/hf_export")
    ap.add_argument("--repo", default=None, help="HF repo id, e.g. <user>/localagent-ultra-tiny-1m")
    ap.add_argument("--token", default=None, help="HF token (else uses HF_TOKEN env / cached login)")
    ap.add_argument("--public", action="store_true", help="create a public repo (default private)")
    ap.add_argument("--push", action="store_true", help="upload to the Hub (requires a token)")
    args = ap.parse_args()

    if args.push and not args.repo:
        raise SystemExit("--push requires --repo <user>/<name>")
    res = export_hf(args.checkpoint, args.out, repo_id=args.repo, token=args.token,
                    private=not args.public, push=args.push)
    if args.push:
        print(f"\n✓ pushed to {res}")
    else:
        has_tok = bool(args.token or os.environ.get("HF_TOKEN"))
        print(f"\n✓ local HF bundle written to {res}/")
        print("  to upload: set a token (HF_TOKEN env or `hf auth login`) and add "
              "--repo <user>/<name> --push" + ("" if has_tok else "   [no token detected]"))


if __name__ == "__main__":
    main()
