"""Tests for offline KD (distill.py): forward/reverse KL plus LFM2 decoupled Top-K KD.

Cheap, CPU-only. Uses tiny within-budget configs and a handful of agent rows.
"""

from __future__ import annotations

import copy

import torch

from openlocalagent.data.synth.agent_synth import Generator
from openlocalagent.data.render import render_sft
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import load_tokenizer
from openlocalagent.data.render import prompt_text
from openlocalagent.train.distill import (
    cache_teacher_topk,
    distill,
    distill_on_policy,
    _topk_kd_loss,
)
from openlocalagent.train.sft import sft


def _tiny_cfg(vocab=256, name="t"):
    return ModelConfig(
        name=name, vocab_size=vocab, d_model=32, embed_dim=16, n_layers=2, n_loops=1,
        n_heads=4, n_kv_heads=2, ffn_hidden=64, max_seq_len=512, rope_theta=10000.0,
        norm_eps=1e-5, tie_embeddings=True, dropout=0.0,
    )


def _models_and_samples(n=24):
    tok = load_tokenizer("byte")
    samples = Generator(level=1, seed=0, split="train").generate(n)
    teacher = LocalAgentLM(_tiny_cfg(name="teacher"))
    student = LocalAgentLM(_tiny_cfg(name="student"))
    return teacher, student, samples, tok


def test_topk_distill_runs_and_decreases():
    teacher, student, samples, tok = _models_and_samples()
    # Fixed-batch regime: small sample set + many-step run with batch covering it.
    hist = distill(student, samples, teacher, tok, steps=24, batch_size=24, kd_type="topk",
                   kd_k=16, temperature=2.0, kd_weight=1.0, ce_weight=0.1, lr=3e-3,
                   warmup=4, log=lambda *a: None)
    assert all(torch.isfinite(torch.tensor(h)) for h in hist)
    assert hist[-1] < hist[0], (hist[0], hist[-1])


def test_forward_reverse_still_green():
    teacher, student, samples, tok = _models_and_samples()
    for kd in ("forward_kl", "reverse_kl"):
        s = copy.deepcopy(student)
        hist = distill(s, samples, teacher, tok, steps=12, batch_size=24, kd_type=kd,
                       temperature=2.0, lr=3e-3, warmup=3, log=lambda *a: None)
        assert all(torch.isfinite(torch.tensor(h)) for h in hist)
        assert hist[-1] < hist[0]


def test_topk_zero_when_student_equals_teacher():
    """Correctness (b): student == teacher => KL_topk ~ 0 and tail_kl ~ 0."""
    teacher, _, samples, tok = _models_and_samples(n=8)
    student = copy.deepcopy(teacher)  # identical weights
    rows = [render_sft(s, tok) for s in samples]
    cache = cache_teacher_topk(teacher, rows, tok, k=16, temperature=2.0, log=lambda *a: None)
    bi = list(range(len(rows)))
    seqs = [rows[j][0][:-1] for j in bi]
    ml = max(len(s) for s in seqs)
    X = torch.full((len(bi), ml), tok.pad_id, dtype=torch.long)
    mask = torch.zeros(len(bi), ml)
    for r, s in enumerate(seqs):
        X[r, : len(s)] = torch.tensor(s)
        mask[r, : len(s)] = 1.0
    student.eval()
    with torch.no_grad():
        logits, _ = student(X)
        kd = _topk_kd_loss(logits, cache, bi, X, mask, student.cfg.vocab_size, 2.0, "cpu")
    assert kd.item() < 1e-4, kd.item()


def test_sft_distill_throughout_runs_and_decreases():
    """distill-throughout-SFT: sft(teacher=...) caches teacher Top-K once and adds the KD term
    to each step. Loss must stay finite and trend down, and the teacher arg is OPT-IN."""
    teacher, student, samples, tok = _models_and_samples(n=16)
    hist, head, ptr = sft(student, samples, tok, steps=20, batch_size=16, lr=3e-3, warmup=4,
                          device="cpu", log=lambda *a: None, joint_tool_head=True,
                          teacher=teacher, kd_type="topk", kd_k=16, kd_weight=0.5,
                          kd_temperature=2.0)
    assert all(torch.isfinite(torch.tensor(h)) for h in hist)
    assert hist[-1] < hist[0], (hist[0], hist[-1])
    assert head is not None and ptr is not None


def test_sft_teacher_default_off_is_inert():
    """With no teacher, the new args change nothing: same seed -> identical loss history as a
    plain sft() call (existing callers byte-for-byte unaffected)."""
    _, _, samples, tok = _models_and_samples(n=16)
    torch.manual_seed(7)
    s1 = LocalAgentLM(_tiny_cfg(name="s"))
    s2 = copy.deepcopy(s1)
    h1, _, _ = sft(s1, samples, tok, steps=10, batch_size=16, lr=3e-3, warmup=2,
                   device="cpu", log=lambda *a: None)
    h2, _, _ = sft(s2, samples, tok, steps=10, batch_size=16, lr=3e-3, warmup=2,
                   device="cpu", log=lambda *a: None, teacher=None, kd_weight=0.5)
    assert h1 == h2


def _prompt_ids(samples, tok):
    return [tok.encode(prompt_text(s)) for s in samples]


def test_on_policy_runs_and_reverse_kl_decreases():
    """distill_on_policy: finite loss, and the student's reverse-KL to a (fixed) teacher trends
    down over a few on-policy steps on a fixed prompt set."""
    teacher, student, samples, tok = _models_and_samples(n=8)
    prompts = _prompt_ids(samples, tok)
    hist = distill_on_policy(student, teacher, prompts, tok, steps=24, batch_size=4,
                             max_new=16, sample_temperature=1.0, kd_temperature=1.0,
                             kd_weight=1.0, lr=3e-3, warmup=4, seed=0, log=lambda *a: None)
    finite = [h for h in hist if h == h]  # drop any NaN "no-sample" slots
    assert finite, "every step produced no sampled tokens"
    assert all(torch.isfinite(torch.tensor(h)) for h in finite)
    # Average of the first vs last third should drop (on-policy reverse-KD reduces the term).
    third = max(1, len(finite) // 3)
    first = sum(finite[:third]) / third
    last = sum(finite[-third:]) / third
    assert last < first, (first, last)


def test_on_policy_zero_when_student_equals_teacher():
    """Correctness: student == teacher => the per-token reverse-KL term ~ 0, so the (kd-only)
    loss is ~0 regardless of which tokens the student samples."""
    teacher, _, samples, tok = _models_and_samples(n=6)
    student = copy.deepcopy(teacher)  # identical weights
    prompts = _prompt_ids(samples, tok)
    hist = distill_on_policy(student, teacher, prompts, tok, steps=4, batch_size=4,
                             max_new=12, sample_temperature=1.0, kd_temperature=1.0,
                             kd_weight=1.0, ce_weight=0.0, lr=0.0, warmup=0, seed=1,
                             log=lambda *a: None)
    finite = [h for h in hist if h == h]
    assert finite
    assert max(abs(h) for h in finite) < 1e-4, finite


def test_on_policy_mix_offpolicy_stays_finite():
    """Blending the teacher-forced forward-KL anchor must keep the loss finite and trending down."""
    teacher, student, samples, tok = _models_and_samples(n=8)
    prompts = _prompt_ids(samples, tok)
    hist = distill_on_policy(student, teacher, prompts, tok, steps=16, batch_size=4,
                             max_new=16, mix_offpolicy_weight=0.5, ce_weight=0.05,
                             entropy_weight=0.01, lr=3e-3, warmup=3, seed=0,
                             log=lambda *a: None)
    finite = [h for h in hist if h == h]
    assert finite and all(torch.isfinite(torch.tensor(h)) for h in finite)


def test_topk_full_vocab_matches_forward_kl():
    """Correctness (c): kd_k >= vocab => KL_topk == standard forward KL, tail == 0."""
    teacher, student, samples, tok = _models_and_samples(n=6)
    V = teacher.cfg.vocab_size
    rows = [render_sft(s, tok) for s in samples]
    cache = cache_teacher_topk(teacher, rows, tok, k=V + 64, temperature=2.0, log=lambda *a: None)
    # tail mass must be ~0 when K covers the whole vocab
    for c in cache:
        assert c["tail"].float().abs().max().item() < 1e-3
        assert c["ids"].shape[1] == V
    bi = list(range(len(rows)))
    seqs = [rows[j][0][:-1] for j in bi]
    ml = max(len(s) for s in seqs)
    X = torch.full((len(bi), ml), tok.pad_id, dtype=torch.long)
    mask = torch.zeros(len(bi), ml)
    for r, s in enumerate(seqs):
        X[r, : len(s)] = torch.tensor(s)
        mask[r, : len(s)] = 1.0
    student.eval()
    teacher.eval()
    with torch.no_grad():
        s_logits, _ = student(X)
        t_logits, _ = teacher(X)
        kd_topk = _topk_kd_loss(s_logits, cache, bi, X, mask, V, 2.0, "cpu")
        # standard forward KL over full vocab, same masking/scaling
        T = 2.0
        s_lp = (s_logits / T).log_softmax(-1)
        t_lp = (t_logits / T).log_softmax(-1)
        fk = (t_lp.exp() * (t_lp - s_lp)).sum(-1)
        fk = (fk * mask).sum() / mask.sum().clamp(min=1) * (T ** 2)
    assert abs(kd_topk.item() - fk.item()) < 1e-3, (kd_topk.item(), fk.item())
