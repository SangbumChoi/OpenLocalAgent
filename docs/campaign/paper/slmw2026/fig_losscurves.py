#!/usr/bin/env python3
"""Matched pretraining-loss traces for the cross-shape initialisation experiment.

Every curve comes from a real 16,000-step run under the same 96M configuration, corpus, seed and
optimisation budget; only initialisation changes. Solid curves are 50-step exponential moving
averages of the per-step training loss, and markers are the held-out validation losses logged
every 2,000 steps. The inset magnifies the second half of training.
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "figures"
DATA = Path(__file__).resolve().parent / "results/analysis/losscurves_perstep.json"
curves = json.load(open(DATA))

INK = "#1A1A1A"
STYLES = {
    "pre-rand-96m": ("random", "#4D4D4D", 1.6),
    "pre-skel-96m": ("projected skeleton", "#0072B2", 1.2),
    "pre-anti-96m": ("projected feed-forward", "#E69F00", 1.2),
    "pre-embed-96m": ("projected embedding", "#CC79A7", 1.2),
}


def smooth(values, span=50):
    out, acc = [], values[0]
    alpha = 2.0 / (span + 1)
    for v in values:
        acc += alpha * (v - acc)
        out.append(acc)
    return out


def colour(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


W, H = 2200, 650
image = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(image)
axis = (150, 45, 1100, 540)
x0, y0, x1, y1 = axis
grid = (210, 210, 210)
ink = colour(INK)


def xmap(step: float, left=x0, right=x1, lo=0, hi=16000) -> float:
    return left + (step - lo) / (hi - lo) * (right - left)


def ymap(value: float, top=y0, bottom=y1, lo=2.2, hi=10.5) -> float:
    a, b, v = math.log(lo), math.log(hi), math.log(value)
    return bottom - (v - a) / (b - a) * (bottom - top)


# Main log-scale axes and grid.
for step in range(0, 16001, 4000):
    x = xmap(step)
    draw.line((x, y0, x, y1), fill=grid, width=1)
    label = f"{step:,}"
    box = draw.textbbox((0, 0), label, font=font(34))
    draw.text((x - (box[2] - box[0]) / 2, y1 + 12), label, fill=ink, font=font(34))
for value in (2.5, 3, 4, 6, 10):
    y = ymap(value)
    draw.line((x0, y, x1, y), fill=grid, width=1)
    label = f"{value:g}"
    box = draw.textbbox((0, 0), label, font=font(34))
    draw.text((x0 - 20 - (box[2] - box[0]), y - 18), label, fill=ink, font=font(34))
draw.line((x0, y0, x0, y1), fill=ink, width=3)
draw.line((x0, y1, x1, y1), fill=ink, width=3)

smoothed = {}
for key, (label, hex_colour, lw) in STYLES.items():
    smoothed[key] = smooth(curves[key]["train"])
    points = [(xmap(i), ymap(v)) for i, v in enumerate(smoothed[key]) if i % 4 == 0]
    draw.line(points, fill=colour(hex_colour), width=max(3, round(lw * 3)), joint="curve")
    for step, value in curves[key]["val"]:
        x, y = xmap(step), ymap(value)
        radius = 10
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=colour(hex_colour), outline="white", width=2)

# Legend in the open upper-right of the full-range panel.
legend_x, legend_y = 585, 58
for index, (key, (label, hex_colour, lw)) in enumerate(STYLES.items()):
    y = legend_y + index * 49
    draw.line((legend_x, y + 18, legend_x + 64, y + 18), fill=colour(hex_colour),
              width=max(5, round(lw * 4)))
    draw.text((legend_x + 78, y), label, fill=ink, font=font(36))

# Second panel: the second half on a linear scale.
ins = (1270, 45, 2130, 540)
ix0, iy0, ix1, iy1 = ins
draw.rectangle(ins, fill="white", outline=ink, width=3)


def ixmap(step: float) -> float:
    return ix0 + (step - 8000) / 8000 * (ix1 - ix0)


def iymap(value: float) -> float:
    return iy1 - (value - 2.30) / (2.75 - 2.30) * (iy1 - iy0)


for step in (8000, 12000, 16000):
    x = ixmap(step)
    draw.line((x, iy0, x, iy1), fill=grid, width=1)
    label = f"{step:,}"
    box = draw.textbbox((0, 0), label, font=font(28))
    draw.text((x - (box[2] - box[0]) / 2, iy1 + 7), label, fill=ink, font=font(28))
for value in (2.3, 2.4, 2.5, 2.6, 2.7):
    y = iymap(value)
    draw.line((ix0, y, ix1, y), fill=grid, width=1)
    draw.text((ix0 - 72, y - 15), f"{value:.1f}", fill=ink, font=font(28))
for key, (_, hex_colour, _) in STYLES.items():
    points = [(ixmap(i), iymap(smoothed[key][i])) for i in range(8000, 16000, 3)]
    draw.line(points, fill=colour(hex_colour), width=3)
    for step, value in curves[key]["val"]:
        if step < 8000:
            continue
        x, y = ixmap(step), iymap(value)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=colour(hex_colour),
                     outline="white", width=2)
draw.text((ix0 + 12, iy0 + 10), "steps 8,000–16,000 (linear)", fill=ink,
          font=font(32, bold=True))

# Axis labels.
xlabel = "pretraining step"
box = draw.textbbox((0, 0), xlabel, font=font(42))
draw.text(((x0 + x1 - (box[2] - box[0])) / 2, 584), xlabel, fill=ink, font=font(42))
zoom_label = "pretraining step (zoom)"
box = draw.textbbox((0, 0), zoom_label, font=font(42))
draw.text(((ix0 + ix1 - (box[2] - box[0])) / 2, 584), zoom_label, fill=ink, font=font(42))
ylabel = "loss (log scale; markers: validation)"
label_layer = Image.new("RGBA", (720, 70), (255, 255, 255, 0))
label_draw = ImageDraw.Draw(label_layer)
label_draw.text((10, 10), ylabel, fill=ink, font=font(38))
label_layer = label_layer.rotate(90, expand=True)
image.paste(label_layer, (18, 20), label_layer)

image.save(OUT / "pf_losses.png", dpi=(300, 300))
print("wrote", OUT / "pf_losses.png")
