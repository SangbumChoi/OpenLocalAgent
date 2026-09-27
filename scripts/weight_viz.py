"""Visualize what *more training data* does to the weights — random init vs synthetic-tool SFT vs
FineWeb-200M pretrain (the GPU run). Saves a 4-panel figure to runs/weight_patterns.png."""
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from localagent.model import LocalAgentLM, ModelConfig

cfg = ModelConfig(**torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")["cfg"])
models = {
    "random init": LocalAgentLM(cfg).state_dict(),
    "synthetic (tool SFT)": torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")["state_dict"],
    "FineWeb 200M tokens": torch.load(
        hf_hub_download("danelcsb/localagent-30m-v2", "pretrain.pt"), map_location="cpu")["state_dict"],
}
col = {"random init": "#999999", "synthetic (tool SFT)": "#1f77b4", "FineWeb 200M tokens": "#d62728"}
nL = cfg.n_layers
lin = [k for k in models["random init"]
       if k.endswith(".weight") and models["random init"][k].ndim == 2 and "norm" not in k]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# A — value distribution of every linear weight (log-y); training sharpens/structures it
for name, sd in models.items():
    w = torch.cat([sd[k].flatten() for k in lin]).float().numpy()
    ax[0, 0].hist(w, bins=240, range=(-0.3, 0.3), histtype="step", log=True, label=name, color=col[name])
    print(f"{name:22} all-weight std={w.std():.4f} kurtosis~{((w-w.mean())**4).mean()/w.var()**2:.1f}",
          flush=True)
ax[0, 0].set_title("weight value distribution (all linear weights)")
ax[0, 0].set_xlabel("weight"); ax[0, 0].legend(fontsize=8)

# B — attention output-proj weight std by depth (how training shapes per-layer scale)
for name, sd in models.items():
    ax[0, 1].plot(range(nL), [sd[f"blocks.{i}.attn.o.weight"].std().item() for i in range(nL)],
                  marker="o", ms=4, label=name, color=col[name])
ax[0, 1].set_title("attention output-proj weight std vs depth")
ax[0, 1].set_xlabel("layer"); ax[0, 1].legend(fontsize=8)

# C — singular-value spectrum of the byte embedding (normalized); training => structured low-rank
for name, sd in models.items():
    s = torch.linalg.svdvals(sd["embed.weight"].float()).numpy()
    ax[1, 0].plot(s / s[0], label=name, color=col[name])
ax[1, 0].set_yscale("log"); ax[1, 0].set_title("byte-embedding singular values (normalized)")
ax[1, 0].set_xlabel("index"); ax[1, 0].legend(fontsize=8)

# D — the FineWeb byte-embedding itself (ASCII 32..127 × first 64 dims): the learned 'pattern'
emb = models["FineWeb 200M tokens"]["embed.weight"].float().numpy()
im = ax[1, 1].imshow(emb[32:128, :64], aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=0.15)
ax[1, 1].set_title("FineWeb byte-embedding (ASCII 32–127 × first 64 dims)")
ax[1, 1].set_xlabel("dim"); ax[1, 1].set_ylabel("byte value")
plt.colorbar(im, ax=ax[1, 1], fraction=0.046)

plt.tight_layout()
plt.savefig("runs/weight_patterns.png", dpi=110)
print("SAVED runs/weight_patterns.png", flush=True)
