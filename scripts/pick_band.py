#!/usr/bin/env python
"""The top and bottom third of layers by measured adaptation, as --layers arguments.

  python scripts/pick_band.py runs/analysis/layer_profiles.json qwen3-06b top
"""
import json
import sys

profiles = json.load(open(sys.argv[1]))
tag, which = sys.argv[2], sys.argv[3]
curve = profiles[tag]["per_layer_mean"]
third = max(1, len(curve) // 3)
ranked = sorted(range(len(curve)), key=lambda i: -curve[i])
if which == "top":
    chosen = ranked[:third]
elif which == "bottom":
    chosen = ranked[-third:]
else:  # random third, seeded so the arm is reproducible
    import random
    rng = random.Random(2026)
    chosen = rng.sample(range(len(curve)), third)
print(",".join(str(i) for i in sorted(chosen)))
