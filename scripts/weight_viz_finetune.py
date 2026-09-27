"""Weight patterns across the training STAGES of the GPU model: FineWeb pretrain -> synthetic SFT ->
GRPO. Shows the weights at each stage AND how much/where the synthetic fine-tune moves them from the
pretrained weights. Saves runs/weight_finetune.png."""
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

REPO = "danelcsb/localagent-30m-v2"


def load(f):
    return torch.load(hf_hub_download(REPO, f), map_location="cpu")["state_dict"]


pre, sftw, grpow = load("pretrain.pt"), load("sft.pt"), load("grpo.pt")
stages = {"FineWeb pretrain": pre, "+ synthetic SFT": sftw, "+ GRPO": grpow}
col = {"FineWeb pretrain": "#d62728", "+ synthetic SFT": "#2ca02c", "+ GRPO": "#9467bd"}
keys = [k for k in pre if k.endswith(".weight") and pre[k].ndim == 2 and "norm" not in k]
nL = max(int(k.split(".")[1]) for k in pre if k.startswith("blocks.")) + 1
block_w = ["attn.q", "attn.k", "attn.v", "attn.o", "ffn.gate", "ffn.up", "ffn.down"]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# A — weight value distribution at each stage (fine-tune barely changes the bulk)
for n, sd in stages.items():
    w = torch.cat([sd[k].flatten() for k in keys]).float().numpy()
    ax[0, 0].hist(w, bins=240, range=(-0.4, 0.4), histtype="step", log=True, label=n, color=col[n])
    print(f"{n:18} std={w.std():.4f}", flush=True)
ax[0, 0].set_title("weight value distribution by stage")
ax[0, 0].set_xlabel("weight"); ax[0, 0].legend(fontsize=8)

# B — FFN gate weight std vs depth at each stage
for n, sd in stages.items():
    ax[0, 1].plot(range(nL), [sd[f"blocks.{i}.ffn.gate.weight"].std().item() for i in range(nL)],
                  marker="o", ms=4, label=n, color=col[n])
ax[0, 1].set_title("FFN gate weight std vs depth")
ax[0, 1].set_xlabel("layer"); ax[0, 1].legend(fontsize=8)

# C — the fine-tune DELTA: how much each stage moved the weights (log-y; tightly peaked at 0)
d_sft = torch.cat([(sftw[k] - pre[k]).flatten() for k in keys]).float().numpy()
d_grpo = torch.cat([(grpow[k] - sftw[k]).flatten() for k in keys]).float().numpy()
ax[1, 0].hist(d_sft, bins=200, range=(-0.06, 0.06), histtype="step", log=True,
              label="SFT − pretrain", color="#2ca02c")
ax[1, 0].hist(d_grpo, bins=200, range=(-0.06, 0.06), histtype="step", log=True,
              label="GRPO − SFT", color="#9467bd")
ax[1, 0].set_title("fine-tune weight CHANGE (Δ from previous stage)")
ax[1, 0].set_xlabel("Δ weight"); ax[1, 0].legend(fontsize=8)
print(f"|Δ| SFT−pretrain: mean={abs(d_sft).mean():.5f}  |Δ| GRPO−SFT: mean={abs(d_grpo).mean():.5f}",
      flush=True)


def rel_change(a, b):
    out = []
    for i in range(nL):
        num = sum((b[f"blocks.{i}.{w}.weight"] - a[f"blocks.{i}.{w}.weight"]).norm().item()**2
                  for w in block_w) ** 0.5
        den = sum(a[f"blocks.{i}.{w}.weight"].norm().item()**2 for w in block_w) ** 0.5
        out.append(num / (den + 1e-9))
    return out


# D — WHERE the synthetic fine-tune changed the most (relative ||Δ|| per layer)
ax[1, 1].plot(range(nL), [r * 100 for r in rel_change(pre, sftw)], marker="o", color="#2ca02c",
              label="SFT vs pretrain")
ax[1, 1].plot(range(nL), [r * 100 for r in rel_change(sftw, grpow)], marker="s", color="#9467bd",
              label="GRPO vs SFT")
ax[1, 1].set_title("relative weight change per layer  (||Δ|| / ||W||, %)")
ax[1, 1].set_xlabel("layer"); ax[1, 1].set_ylabel("% changed"); ax[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("runs/weight_finetune.png", dpi=110)
print("SAVED runs/weight_finetune.png", flush=True)
