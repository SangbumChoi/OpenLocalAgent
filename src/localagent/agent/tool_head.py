"""Tool-selection head (ARCHITECTURE_IDEAS §2a, dual-head — tool side).

Generic argument *grounding* works (agent/constrained.py), but a ~1M byte model's intrinsic tool
*selection* is weak — per-tool trigger phrases used to mask that. The fix is a tiny linear
classifier over the prompt's final hidden state that picks the tool (or "text"), trained with a
clean label signal. Cheap to train (a linear probe on frozen features) and accurate.

Selection (this head) + grounding (constrained.py) = a trigger-free, schema-driven decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.model.tokenizer import ASSISTANT, USER

CLASSES = ["get_weather", "calculator", "web_search", "planner",
           "define", "play_music", "get_news",
           "read_file", "write_file", "grep_search", "run_command", "git_commit", "run_tests",
           "set_reminder", "set_timer",
           "calendar_event", "send_email", "open_url", "notion_write", "slack_send", "jira_issue",
           # computer-use family
           "screenshot", "click", "double_click", "type_text", "key_press", "scroll", "drag",
           "wait", "move_cursor", "open_app",
           # modern dev / agentic tools
           "run_python", "edit_file", "apply_patch", "http_request", "sql_query", "list_dir",
           "find_files", "git_diff", "git_status", "install_package", "kill_process",
           "read_clipboard", "write_clipboard", "download_file", "unzip", "env_get", "make_dir",
           "list_processes", "docker_run",
           "text"]

# Public browser datasets name grounded actions explicitly; the legacy 51-way classifier uses
# the equivalent computer-use classes.  Keep this alias only for auxiliary head supervision —
# exported dense dispatch still preserves the public ``web_*`` tool names.
TOOL_ALIASES = {"web_click": "click", "web_type": "type_text", "web_select": "click"}


def canonical_tool_name(name: str) -> str:
    return TOOL_ALIASES.get(name, name)


def label_of(sample) -> str:
    return canonical_tool_name(sample.ref_name) if sample.kind == "tool" else "text"


@torch.no_grad()
def _feat(model, tok, prompt: str, device, *, framed: bool = False) -> torch.Tensor:
    """Return the final prompt feature for raw user text or an already-framed history.

    Training trajectories pass a history that already ends in ``ASSISTANT``. Keeping that explicit
    avoids silently wrapping it in a second ``USER ... ASSISTANT`` pair. Left truncation matches
    the model's inference context window and is inert for legacy single-turn prompts that fit.
    """

    text = prompt if framed else f"{USER}{prompt}{ASSISTANT}"
    encoded = tok.encode(text)[-model.cfg.max_seq_len :]
    ids = torch.tensor([encoded], dtype=torch.long, device=device)
    _, feats = model(ids, return_hidden=True)
    return feats[0, -1]  # final prompt-token features (d_model)


class ToolHead(nn.Module):
    def __init__(self, d_model: int, classes: list[str] = CLASSES):
        super().__init__()
        self.classes = classes
        self.fc = nn.Linear(d_model, len(classes))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)

    def predict(self, model, tok, prompt: str, device="cpu") -> str:
        feat = _feat(model, tok, prompt, device)
        return self.classes[int(self(feat).argmax(-1))]


def train_tool_head(model, samples, tok, *, steps=300, batch_size=64, lr=5e-3, device="cpu",
                    log=lambda *a: None) -> ToolHead:
    model.eval()
    head = ToolHead(model.cfg.d_model).to(device)
    # cache frozen features + labels once (linear probe)
    feats = torch.stack(
        [
            _feat(model, tok, s.prompt, device, framed=bool(getattr(s, "framed", False)))
            for s in samples
        ]
    )
    labels = torch.tensor([CLASSES.index(label_of(s)) for s in samples], device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    import random
    rng = random.Random(0)
    n = len(samples)
    for step in range(steps):
        idx = torch.tensor([rng.randrange(n) for _ in range(batch_size)], device=device)
        loss = F.cross_entropy(head(feats[idx]), labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            acc = (head(feats).argmax(-1) == labels).float().mean().item()
            log(f"  [tool-head] step {step}/{steps} loss {loss.item():.3f} train-acc {acc:.3f}")
    return head
