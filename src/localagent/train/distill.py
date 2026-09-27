"""Distillation (Phase 6, implemented): offline logit KD from a larger teacher into the student.

Both models are byte-level (vocab 256), so we distill the **next-byte distribution** on the agent
data. The teacher's soft targets are cached once (batched), then the student trains against them —
no teacher forward in the inner loop, which makes it affordable on CPU.

  forward_kl : KL(teacher || student)  — standard KD (match the teacher everywhere).
  reverse_kl : KL(student || teacher)  — mode-seeking (MiniLLM); good for generation.
  topk       : LFM2 "decoupled, tempered Top-K KD" — only cache the teacher's Top-K tokens
               plus the off-Top-K tail mass, then match (a) the renormalized distribution over
               the Top-K support and (b) the in-vs-out-of-support mass split. Avoids the
               support-mismatch instability of full-vocab KD and keeps the cache to K/pos.

The KD is applied on the assistant/tool-call spans (where the label mask is active), optionally
mixed with a little ground-truth CE.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from localagent.data.render import IGNORE, render_sft
from localagent.train.loop import cosine_lr, set_lr


@torch.no_grad()
def cache_teacher(teacher, rows, tok, device="cpu", temperature=2.0, batch_size=16, log=print):
    """Cache teacher log-probs (at `temperature`) for each row's next-token positions."""
    teacher.eval().to(device)
    cache = [None] * len(rows)
    for i in range(0, len(rows), batch_size):
        seqs = [rows[j][0][:-1] for j in range(i, min(i + batch_size, len(rows)))]
        ml = max(len(s) for s in seqs)
        X = torch.full((len(seqs), ml), tok.pad_id, dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            X[r, : len(s)] = torch.tensor(s, device=device)
        logits, _ = teacher(X)
        lp = (logits / temperature).log_softmax(-1)
        for r, s in enumerate(seqs):
            cache[i + r] = lp[r, : len(s)].half().cpu()   # (len, V)
        if i % (batch_size * 10) == 0:
            log(f"  [cache] {i}/{len(rows)}")
    return cache


@torch.no_grad()
def cache_teacher_topk(teacher, rows, tok, device="cpu", temperature=2.0, k=16, batch_size=16,
                       log=print):
    """Cache the teacher's tempered Top-K targets per next-token position (LFM2 KD).

    For each position we store, in a compact per-row dict of tensors:
      ids  : (len, K) long  — the teacher's Top-K token ids (by tempered prob)
      p    : (len, K) half  — teacher tempered probs at those ids (softmax over the FULL vocab)
      tail : (len,)   half  — 1 - sum(top-K probs), the mass outside the Top-K support
    K is clamped to the vocab size, so for byte models (vocab 256) it can be the full vocab.
    """
    teacher.eval().to(device)
    V = teacher.cfg.vocab_size
    k = min(k, V)
    cache = [None] * len(rows)
    for i in range(0, len(rows), batch_size):
        seqs = [rows[j][0][:-1] for j in range(i, min(i + batch_size, len(rows)))]
        ml = max(len(s) for s in seqs)
        X = torch.full((len(seqs), ml), tok.pad_id, dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            X[r, : len(s)] = torch.tensor(s, device=device)
        logits, _ = teacher(X)
        probs = (logits / temperature).softmax(-1)            # (B, ml, V) over FULL vocab
        topp, topi = probs.topk(k, dim=-1)                    # (B, ml, K)
        tail = (1.0 - topp.sum(-1)).clamp(min=0.0)            # (B, ml)
        for r, s in enumerate(seqs):
            n = len(s)
            cache[i + r] = {
                "ids": topi[r, :n].cpu(),
                "p": topp[r, :n].half().cpu(),
                "tail": tail[r, :n].half().cpu(),
            }
        if i % (batch_size * 10) == 0:
            log(f"  [cache-topk] {i}/{len(rows)}")
    return cache


def _topk_kd_loss(logits, cache, bi, X, mask, V, temperature, device):
    """Decoupled tempered Top-K KD term (LFM2), summed over masked positions then averaged.

    Per position p, with teacher Top-K support S (ids), teacher tempered probs t_S over S, and
    teacher tail mass m_tail = 1 - sum(t_S):
      KL_topk = KL(t_hat_S || s_hat_S), where t_hat / s_hat renormalize t_S / s_S over S only.
      tail_kl = binary KL( [sum(t_S), m_tail]  ||  [sum(s_S), 1 - sum(s_S)] ).
    Returns (KL_topk + tail_kl) * T^2 averaged over masked positions.
    """
    B, ml = X.shape
    k = cache[bi[0]]["ids"].shape[1]
    ids = torch.zeros(B, ml, k, dtype=torch.long, device=device)
    t_p = torch.zeros(B, ml, k, device=device)
    t_tail = torch.zeros(B, ml, device=device)
    for r, j in enumerate(bi):
        c = cache[j]
        n = c["ids"].shape[0]
        ids[r, :n] = c["ids"].to(device)
        t_p[r, :n] = c["p"].float().to(device)
        t_tail[r, :n] = c["tail"].float().to(device)
    s_probs = (logits / temperature).softmax(-1)              # (B, ml, V)
    s_p = s_probs.gather(-1, ids)                             # (B, ml, K) student probs on S
    eps = 1e-9
    # (1) Top-K matching: renormalize both over S, KL(teacher_S || student_S).
    t_in = t_p.sum(-1).clamp(min=eps)                         # teacher mass on S
    s_in = s_p.sum(-1).clamp(min=eps)                         # student mass on S
    t_hat = t_p / t_in.unsqueeze(-1)
    s_hat = s_p / s_in.unsqueeze(-1)
    kl_topk = (t_hat * ((t_hat + eps).log() - (s_hat + eps).log())).sum(-1)
    # (2) Tail-mass matching: binary KL between (in_S, out_S) splits. The teacher's out-of-S
    # mass is the cached `tail` (= 1 - sum(top-K)); the student's is computed live from s_in.
    t_in_c = (1.0 - t_tail).clamp(min=eps, max=1.0)           # teacher in-S mass (from cache)
    t_out = t_tail.clamp(min=eps, max=1.0)
    s_in_c = s_in.clamp(min=eps, max=1.0)
    s_out = (1.0 - s_in_c).clamp(min=eps)
    tail_kl = (t_in_c * ((t_in_c + eps).log() - (s_in_c + eps).log())
               + t_out * ((t_out + eps).log() - (s_out + eps).log()))
    kd = (kl_topk + tail_kl) * mask
    return kd.sum() / mask.sum().clamp(min=1) * (temperature ** 2)


def distill(student, samples, teacher, tok, *, steps=400, batch_size=24, kd_type="forward_kl",
            temperature=2.0, kd_weight=1.0, ce_weight=0.1, kd_k=16, lr=1e-3, warmup=30,
            device="cpu", log=print):
    """Distill `teacher` (a trained model) into `student` via cached logit KD on agent samples."""
    rows = [render_sft(s, tok) for s in samples]
    is_topk = kd_type == "topk"
    log("caching teacher soft targets ..." + (" (top-k)" if is_topk else ""))
    if is_topk:
        cache = cache_teacher_topk(teacher, rows, tok, device, temperature, k=kd_k, log=log)
    else:
        cache = cache_teacher(teacher, rows, tok, device, temperature, log=log)
    student.train().to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95))
    rng = random.Random(0)
    V = student.cfg.vocab_size
    hist = []
    for step in range(steps):
        set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
        bi = [rng.randrange(len(rows)) for _ in range(batch_size)]
        seqs = [rows[j][0][:-1] for j in bi]
        labs = [rows[j][1][1:] for j in bi]
        ml = max(len(s) for s in seqs)
        X = torch.full((len(bi), ml), tok.pad_id, dtype=torch.long, device=device)
        Y = torch.full((len(bi), ml), IGNORE, dtype=torch.long, device=device)
        for r, j in enumerate(bi):
            X[r, : len(seqs[r])] = torch.tensor(seqs[r], device=device)
            Y[r, : len(labs[r])] = torch.tensor(labs[r], device=device)
        logits, _ = student(X)
        mask = (Y != IGNORE).float()
        if is_topk:
            kd = _topk_kd_loss(logits, cache, bi, X, mask, V, temperature, device)
        else:
            TL = torch.zeros(len(bi), ml, V, device=device)
            for r, j in enumerate(bi):
                tc = cache[j]
                TL[r, : tc.shape[0]] = tc.float().to(device)
            s_lp = (logits / temperature).log_softmax(-1)
            if kd_type == "reverse_kl":
                kd = (s_lp.exp() * (s_lp - TL)).sum(-1)
            else:  # forward_kl
                kd = (TL.exp() * (TL - s_lp)).sum(-1)
            kd = (kd * mask).sum() / mask.sum().clamp(min=1) * (temperature ** 2)
        loss = kd_weight * kd
        if ce_weight > 0:
            loss = loss + ce_weight * F.cross_entropy(
                logits.reshape(-1, V), Y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            log(f"  [distill] step {step}/{steps}  loss {loss.item():.3f}")
    return hist


@torch.no_grad()
def _sample_student_batch(student, prompt_ids, tok, *, max_new, temperature, device, rng):
    """Sample one continuation per prompt from the STUDENT (eval, no-grad), via the KV cache.

    Returns, per prompt, the list of *generated* token ids (EOS dropped, capped at `max_new`).
    Sampling is multinomial at `temperature` (set temperature<=0 for greedy). Each prompt is
    decoded independently to keep the incremental-decode path simple and CPU-cheap on short seqs.
    """
    was_training = student.training
    student.eval()
    gens = []
    for pids in prompt_ids:
        ids = list(pids)
        caches = [None] * student.n_cache_slots()
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _, caches = student(x, pos=0, caches=caches)
        pos = len(ids)
        gen = []
        for _ in range(max_new):
            row = logits[0, -1]
            if temperature and temperature > 0:
                probs = F.softmax(row / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1, generator=rng))
            else:
                nxt = int(row.argmax())
            if nxt == tok.eos_id:
                break
            gen.append(nxt)
            step = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, _, caches = student(step, pos=pos, caches=caches)
            pos += 1
        gens.append(gen)
    if was_training:
        student.train()
    return gens


def distill_on_policy(student, teacher, prompts, tok, *, steps=200, batch_size=4,
                      max_new=48, sample_temperature=1.0, kd_temperature=1.0,
                      kd_weight=1.0, ce_weight=0.0, entropy_weight=0.0,
                      mix_offpolicy_weight=0.0, lr=1e-3, warmup=20, grad_clip=1.0,
                      seed=0, device="cpu", log=print):
    """On-policy reverse-KL distillation (MiniLLM / on-policy-KD; ARCHITECTURE_DEBATE axis 4).

    Unlike the off-policy :func:`distill` (teacher-forced on cached teacher targets), here each
    step distills on the STUDENT's own freshly-sampled trajectories, which is what attacks
    exposure bias (O(eps*T^2) -> O(eps*T)) and targets the stuck free-rollout metric. Nothing is
    cached across steps — trajectories are regenerated every step from the current student.

    Objective (token-level reverse KL on student-sampled tokens). For each sampled continuation
    position t we minimise the per-token reverse-KL integrand

        sum_v p_student(v) * (log p_student(v) - log p_teacher(v))    [full reverse KL]

    evaluated at the (tempered) student and teacher next-token distributions over the FULL vocab,
    averaged over all sampled (non-prompt) positions in the batch. This is the standard mode-
    seeking reverse-KL KD term; because the positions themselves are drawn from the student's own
    rollout distribution, the expectation is taken on-policy. We use the analytic per-position KL
    (not a single-sample REINFORCE estimate) so no baseline is needed and the term is low-variance;
    the on-policy aspect comes from *where* (which states) we evaluate it. By construction, if
    student == teacher the per-token reverse KL is 0 (asserted in tests).

    Stabilisers (all default-off):
      * ``ce_weight``        — ground-truth CE on the sampled tokens (treat student sample as the
                               label) to keep the student's own modes sharp.
      * ``entropy_weight``   — adds +H(student) (i.e. subtracts entropy from the loss) to *discourage*
                               collapse; set >0 to keep the sampling distribution from degenerating.
      * ``mix_offpolicy_weight`` — blends in the teacher-forced forward-KL term on the *prompt+sample*
                               sequence (cheap, no caching) for extra stability.

    `prompts` is a list of prompt token-id lists (e.g. ``tok.encode(prompt_text(s))``). Returns the
    per-step loss history.
    """
    student.to(device)
    teacher.eval().to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95))
    py_rng = random.Random(seed)
    gen_rng = torch.Generator(device=device)
    gen_rng.manual_seed(seed)
    V = student.cfg.vocab_size
    T = kd_temperature
    hist = []
    for step in range(steps):
        set_lr(opt, cosine_lr(step, steps, lr, warmup, 0.1))
        bi = [py_rng.randrange(len(prompts)) for _ in range(batch_size)]
        batch_prompts = [list(prompts[j]) for j in bi]
        gens = _sample_student_batch(
            student, batch_prompts, tok, max_new=max_new, temperature=sample_temperature,
            device=device, rng=gen_rng)
        # Build a padded (prompt + sampled-continuation) batch; mask = sampled (non-prompt) positions.
        seqs = [p + g for p, g in zip(batch_prompts, gens)]
        plen = [len(p) for p in batch_prompts]
        # Drop rows whose student produced no tokens (nothing to learn on this step's slot).
        keep = [r for r in range(len(seqs)) if len(seqs[r]) - plen[r] > 0 and len(seqs[r]) >= 2]
        if not keep:
            hist.append(float("nan"))
            if step % max(1, steps // 6) == 0 or step == steps - 1:
                log(f"  [on-policy] step {step}/{steps}  (no sampled tokens)")
            continue
        seqs = [seqs[r] for r in keep]
        plen = [plen[r] for r in keep]
        ml = max(len(s) for s in seqs)
        X = torch.full((len(seqs), ml), tok.pad_id, dtype=torch.long, device=device)
        # mask over positions in the *input* X that PREDICT a sampled token: a position i predicts
        # token i+1, so a sampled token at index j (j >= plen) is predicted from input index j-1.
        mask = torch.zeros(len(seqs), ml, device=device)
        smpl = torch.full((len(seqs), ml), IGNORE, dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            X[r, : len(s)] = torch.tensor(s, device=device)
            for j in range(plen[r], len(s)):
                mask[r, j - 1] = 1.0
                smpl[r, j - 1] = s[j]
        student.train()
        logits, _ = student(X)
        with torch.no_grad():
            t_logits, _ = teacher(X)
        s_lp = (logits / T).log_softmax(-1)            # (B, ml, V)
        t_lp = (t_logits / T).log_softmax(-1)
        s_p = s_lp.exp()
        # Per-position full reverse KL: sum_v p_s (log p_s - log p_t).
        rkl = (s_p * (s_lp - t_lp)).sum(-1)            # (B, ml)
        denom = mask.sum().clamp(min=1)
        kd = (rkl * mask).sum() / denom * (T ** 2)
        loss = kd_weight * kd
        if ce_weight > 0:
            loss = loss + ce_weight * F.cross_entropy(
                logits.reshape(-1, V), smpl.reshape(-1), ignore_index=IGNORE)
        if entropy_weight > 0:
            # +H(student) added so it *reduces* the loss => encourages keeping entropy up.
            ent = -(s_p * s_lp).sum(-1)
            loss = loss - entropy_weight * (ent * mask).sum() / denom
        if mix_offpolicy_weight > 0:
            # Teacher-forced forward KL over the same masked positions (extra stability anchor).
            fk = (t_lp.exp() * (t_lp - s_lp)).sum(-1)
            loss = loss + mix_offpolicy_weight * (fk * mask).sum() / denom * (T ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        opt.step()
        hist.append(loss.item())
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            log(f"  [on-policy] step {step}/{steps}  loss {loss.item():.4f}  kd {kd.item():.4f}")
    return hist


def run(config_path: str) -> None:
    raise NotImplementedError("Use scripts/distill_demo.py — distill() is called in-process there")
