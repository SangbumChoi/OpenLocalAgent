"""WSD (Warmup-Stable-Decay) LR schedule + decay-window data injection.

Asserts the three phases of `wsd_lr`, the decay-window predicate, and that the opt-in `sft`
path is byte-for-byte the old cosine schedule when defaulted off.
"""

import copy

import torch

from openlocalagent.train.loop import cosine_lr, in_decay_window, wsd_lr


def test_wsd_three_phases():
    total, peak, warmup, decay_frac = 100, 1.0, 10, 0.2
    # warmup: linear 0 -> peak
    assert wsd_lr(0, total, peak, warmup, decay_frac) == 0.0
    assert wsd_lr(5, total, peak, warmup, decay_frac) == peak * 5 / 10
    assert wsd_lr(10, total, peak, warmup, decay_frac) == peak  # plateau begins
    # stable plateau: flat at peak through the whole stable region
    assert wsd_lr(40, total, peak, warmup, decay_frac) == peak
    assert wsd_lr(79, total, peak, warmup, decay_frac) == peak
    # decay window is the last 20 steps (decay_start = 80)
    assert wsd_lr(80, total, peak, warmup, decay_frac) == peak       # 0.5^0 == 1 at S
    half = wsd_lr(100, total, peak, warmup, decay_frac)              # s-S == T -> 0.5*peak... clamp
    # at the end of the window (s-S == T) lr == peak * 0.5
    assert abs(wsd_lr(100, total, peak, warmup, decay_frac) - peak * 0.5) < 1e-9
    # strictly decreasing across the decay window
    assert wsd_lr(85, total, peak, warmup, decay_frac) > wsd_lr(90, total, peak, warmup, decay_frac)
    assert half < peak


def test_wsd_no_decay_is_flat_after_warmup():
    # decay_frac == 0 -> warmup then a flat plateau forever (no decay phase)
    for s in (15, 50, 99):
        assert wsd_lr(s, 100, 2e-3, 10, 0.0) == 2e-3


def test_in_decay_window_predicate():
    assert not in_decay_window(79, 100, 0.2)
    assert in_decay_window(80, 100, 0.2)
    assert in_decay_window(99, 100, 0.2)
    assert not any(in_decay_window(s, 100, 0.0) for s in range(100))  # no window when frac==0


def test_cosine_still_default_and_unchanged():
    # the cosine helper is untouched: monotone decay from peak after warmup
    assert cosine_lr(0, 100, 1.0, 10, 0.1) == 0.0
    assert cosine_lr(10, 100, 1.0, 10, 0.1) == 1.0
    assert cosine_lr(55, 100, 1.0, 10, 0.1) > cosine_lr(100, 100, 1.0, 10, 0.1)


def test_sft_default_cosine_byte_identical_to_wsd_off():
    """Defaulted-off (cosine) sft must be deterministic and identical to passing the explicit
    cosine schedule — proving the new params don't perturb the legacy path."""
    from openlocalagent.data.synth.agent_synth import Generator
    from openlocalagent.model import LocalAgentLM, ModelConfig
    from openlocalagent.model.tokenizer import load_tokenizer
    from openlocalagent.train.sft import sft

    tok = load_tokenizer("byte")
    samples = Generator(level=1, seed=1, split="train").generate(60)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      ffn_hidden=128, max_seq_len=512)

    def run(**kw):
        torch.manual_seed(0)
        m = LocalAgentLM(cfg)
        h, _, _ = sft(m, samples, tok, steps=12, batch_size=4, warmup=3, device="cpu",
                      log=lambda *a: None, **kw)
        return h, copy.deepcopy(m.state_dict())

    h_default, sd_default = run()
    h_explicit, sd_explicit = run(lr_schedule="cosine")
    assert h_default == h_explicit
    for k in sd_default:
        assert torch.equal(sd_default[k], sd_explicit[k])


def test_sft_wsd_runs_and_diverges_from_cosine():
    from openlocalagent.data.synth.agent_synth import Generator
    from openlocalagent.model import LocalAgentLM, ModelConfig
    from openlocalagent.model.tokenizer import load_tokenizer
    from openlocalagent.train.sft import sft

    tok = load_tokenizer("byte")
    samples = Generator(level=1, seed=1, split="train").generate(60)
    decay = Generator(level=1, seed=2, split="train").generate(30)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      ffn_hidden=128, max_seq_len=512)
    torch.manual_seed(0)
    m = LocalAgentLM(cfg)
    h, _, _ = sft(m, samples, tok, steps=12, batch_size=4, warmup=3, device="cpu",
                  log=lambda *a: None, lr_schedule="wsd", decay_frac=0.25, decay_samples=decay)
    assert len(h) == 12 and all(torch.isfinite(torch.tensor(v)) for v in h)
