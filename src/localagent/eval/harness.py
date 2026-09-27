"""Eval harness (Phase 5, implemented): generate greedily, score per group.

Tool samples  -> AST match (name + normalized args) via the tool-call parser.
Text samples  -> exact string match (and must NOT emit a tool call).
Groups: tool_call (weather+calc), web_search, planner, text. These are the categories the
flywheel drives toward 100%.
"""

from __future__ import annotations

import json
from collections import defaultdict

from localagent.agent.parser import extract_tool_calls
from localagent.data.schema import ToolCall
from localagent.eval.tool_eval import match_calls
from localagent.inference.generate import generate


def _correct(sample, gen_text: str) -> bool:
    if sample.kind == "tool":
        if getattr(sample, "calls", None):            # parallel: match the whole set of calls
            gold = [ToolCall(**c) for c in sample.calls]
        else:
            gold = [ToolCall(**json.loads(sample.target))]
        return match_calls(extract_tool_calls(gen_text), gold)
    # text: exact match, and no spurious tool call
    return gen_text.strip() == sample.target.strip() and not extract_tool_calls(gen_text)


def evaluate(model, samples, tok, device="cpu", max_new_tokens=96) -> dict:
    by_group = defaultdict(lambda: [0, 0])   # group -> [correct, total]
    by_cat = defaultdict(lambda: [0, 0])
    n_correct = 0
    for s in samples:
        from localagent.data.render import prompt_text
        gen, _ = generate(model, tok, prompt_text(s), max_new_tokens=max_new_tokens, temperature=0.0)
        ok = _correct(s, gen)
        n_correct += ok
        by_group[s.group][0] += ok
        by_group[s.group][1] += 1
        by_cat[s.category][0] += ok
        by_cat[s.category][1] += 1
    groups = {g: c / t for g, (c, t) in by_group.items()}
    cats = {g: c / t for g, (c, t) in by_cat.items()}
    return {
        "overall": n_correct / len(samples),
        "groups": groups,
        "categories": cats,
        "n": len(samples),
    }


def evaluate_incontext(model, samples, tok, tools, device="cpu", k=8, max_new_tokens=96,
                       retriever=None, seed=0) -> dict:
    """Generative in-context eval — the *generable* pipeline. The candidate tool catalog is rendered
    into the prompt and the LM **free-generates** the full `tool(args)` (AST-matched). No tool head,
    no pointer copy. Two settings, plus retrieval recall:

      gen_acc        — gold tool force-included in the candidate set (isolates generation skill given
                       the right tools are visible).
      gen_acc_e2e    — candidates are the retriever's top-k only (realistic: retrieval may miss gold).
      recall_at_k    — fraction of tool samples whose gold tool is in the retriever's top-k.

    Adding a tool here needs zero retraining — it's one more catalog line."""
    import random

    from localagent.agent.incontext import build_candidates, grounded_prompt
    from localagent.agent.retriever import ToolRetriever
    from localagent.agent.routes import route_of_sample

    by_name = {t.name: t for t in tools}
    retriever = retriever or ToolRetriever(tools)
    rng = random.Random(seed)
    by_route = defaultdict(lambda: [0, 0, 0])   # route -> [gen_ok(gold-in), gen_ok(e2e), total]
    gen_hit = e2e_hit = rec_hit = rec_tot = 0
    for s in samples:
        gold = s.ref_name if s.kind == "tool" else "text"
        cand_g = build_candidates(s.prompt, gold, retriever, by_name, k=k, include_gold=True, rng=rng)
        cand_r = build_candidates(s.prompt, gold, retriever, by_name, k=k, include_gold=False, rng=rng)
        g_ok = _correct(s, generate(model, tok, grounded_prompt(s.prompt, cand_g),
                                    max_new_tokens=max_new_tokens, temperature=0.0)[0])
        e_ok = _correct(s, generate(model, tok, grounded_prompt(s.prompt, cand_r),
                                    max_new_tokens=max_new_tokens, temperature=0.0)[0])
        gen_hit += g_ok
        e2e_hit += e_ok
        if s.kind == "tool":
            rec_tot += 1
            rec_hit += gold in {t.name for t in cand_r}
        r = route_of_sample(s)
        b = by_route[r]
        b[0] += g_ok
        b[1] += e_ok
        b[2] += 1
    n = len(samples)
    return {
        "gen_acc": gen_hit / max(1, n),
        "gen_acc_e2e": e2e_hit / max(1, n),
        "recall_at_k": rec_hit / max(1, rec_tot),
        "by_route": {r: {"gen": b[0] / b[2], "e2e": b[1] / b[2], "n": b[2]}
                     for r, b in sorted(by_route.items())},
        "n": n, "k": k,
    }


def evaluate_hybrid(model, samples, tok, tools, device="cpu", *, retriever=None, route_head=None,
                    ptr_head=None, selector=None, top_m=1, k=8) -> dict:
    """Eval the *generable* hybrid decode (`constrained.hybrid_decode`): trained-dense-selector OR
    retrieval selection + pointer-copy args + optional route gate — NO fixed-N classifier. Reports
    overall + per route."""
    from localagent.agent.constrained import hybrid_decode
    from localagent.agent.retriever import ToolRetriever
    from localagent.agent.routes import route_of_sample

    retriever = retriever or ToolRetriever(tools)
    by_route = defaultdict(lambda: [0, 0])
    n_correct = 0
    for s in samples:
        out = hybrid_decode(model, tok, s.prompt, tools, device=device, retriever=retriever,
                            route_head=route_head, ptr_head=ptr_head, selector=selector,
                            top_m=top_m, k=k)
        ok = _correct(s, out)
        n_correct += ok
        r = route_of_sample(s)
        by_route[r][0] += ok
        by_route[r][1] += 1
    return {
        "overall": n_correct / max(1, len(samples)),
        "by_route": {r: {"acc": c / t, "n": t} for r, (c, t) in sorted(by_route.items())},
        "n": len(samples), "k": k,
    }


def evaluate_grounded(model, samples, tok, tools, device="cpu", tool_head=None, ptr_head=None) -> dict:
    """Eval with grounded constrained decoding (the deployed decoder). A trained `tool_head` does
    tool selection and a `ptr_head` fills arguments via learned copy spans; otherwise heuristic
    selection/extraction is used."""
    from collections import defaultdict

    from localagent.agent.constrained import grounded_decode, grounded_decode_parallel
    by_group = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    n_correct = 0
    for s in samples:
        if getattr(s, "calls", None) and len(s.calls) > 1:   # parallel two-call turn
            out = grounded_decode_parallel(model, tok, s.prompt, tools, device=device,
                                           tool_head=tool_head, ptr_head=ptr_head)
        else:
            out = grounded_decode(model, tok, s.prompt, tools, device=device,
                                  tool_head=tool_head, ptr_head=ptr_head)
        ok = _correct(s, out)
        n_correct += ok
        by_group[s.group][0] += ok
        by_group[s.group][1] += 1
        by_cat[s.category][0] += ok
        by_cat[s.category][1] += 1
    return {
        "overall": n_correct / len(samples),
        "groups": {g: c / t for g, (c, t) in by_group.items()},
        "categories": {g: c / t for g, (c, t) in by_cat.items()},
        "n": len(samples),
    }


def evaluate_routed(model, samples, tok, route_head, device="cpu", max_new_tokens=96) -> dict:
    """Route-based eval (the fixed pipeline). Two decoupled signals, bucketed by the 5 routes:

      route_acc — the small 5-way route head picks the correct modality from the prompt's final
                  hidden state. Stable + portable: unaffected by how many concrete tools exist.
      gen_acc   — the LM *free-generates* the specific `tool(args)` as text, AST-matched against
                  gold (the portable, deployable number — no closed-set classifier involved).

    This replaces "51-way head selects the exact tool": selection is now route (head) + concrete
    call (generation), so adding a tool never reshapes the head."""
    import torch

    from localagent.agent.routes import ROUTE_INDEX, ROUTES, route_of_sample
    from localagent.agent.tool_head import _feat
    from localagent.data.render import prompt_text

    by_route = defaultdict(lambda: [0, 0, 0])   # route -> [route_correct, gen_correct, total]
    route_hit = gen_hit = 0
    for s in samples:
        gold_route = route_of_sample(s)
        with torch.no_grad():
            feat = _feat(model, tok, s.prompt, device)
            pred_route = ROUTES[int(route_head(feat).argmax(-1))]
        gen, _ = generate(model, tok, prompt_text(s), max_new_tokens=max_new_tokens, temperature=0.0)
        r_ok = pred_route == gold_route
        g_ok = _correct(s, gen)
        route_hit += r_ok
        gen_hit += g_ok
        b = by_route[gold_route]
        b[0] += r_ok
        b[1] += g_ok
        b[2] += 1
    n = len(samples)
    return {
        "route_acc": route_hit / max(1, n),
        "gen_acc": gen_hit / max(1, n),
        "by_route": {r: {"route": b[0] / b[2], "gen": b[1] / b[2], "n": b[2]}
                     for r, b in sorted(by_route.items(), key=lambda kv: ROUTE_INDEX.get(kv[0], 9))},
        "n": n,
    }


def multi_turn_eval(model, episodes, tok, tools, device="cpu", tool_head=None, ptr_head=None) -> dict:
    """Replay each episode; at every assistant *tool-call* turn, decode the next action over the
    full history (so follow-up args can be grounded in earlier tool responses) and AST-match it
    against the gold call. Reports per-step and whole-episode accuracy."""
    from localagent.agent.constrained import grounded_decode
    from localagent.agent.parser import extract_tool_calls
    from localagent.data.render import history_text
    from localagent.data.schema import Role
    from localagent.eval.tool_eval import match_calls
    from localagent.model.tokenizer import ASSISTANT

    step_ok = step_tot = ep_ok = ep_tot = 0
    for conv in episodes:
        all_ok, has_step = True, False
        for i, m in enumerate(conv.messages):
            if m.role != Role.assistant or not m.tool_calls:
                continue
            has_step = True
            ctx = history_text(conv.messages[:i]) + ASSISTANT
            out = grounded_decode(model, tok, ctx, tools, device=device, tool_head=tool_head,
                                  ptr_head=ptr_head, framed=True)
            pred = extract_tool_calls(out)
            ok = bool(pred) and match_calls([pred[0]], [m.tool_calls[0]])
            step_ok += ok
            step_tot += 1
            all_ok = all_ok and ok
        if has_step:
            ep_tot += 1
            ep_ok += all_ok
    return {"step_acc": step_ok / max(1, step_tot), "episode_acc": ep_ok / max(1, ep_tot),
            "steps": step_tot, "episodes": ep_tot}


def _first_user_query(conv) -> str:
    """The episode's initial user request — the planner's only input (matches how multi_turn_eval
    starts each episode from the first user turn)."""
    from localagent.data.schema import Role
    for m in conv.messages:
        if m.role == Role.user:
            return m.content
    return ""


def plan_eval(model, tok, tools, episodes, *, tool_head, ptr_head, max_steps=4, device="cpu") -> dict:
    """Free-run PLANNER eval (stage 3) over held-out plan episodes.

    For each episode we take only the first user query, call the learned ``plan_rollout`` (which
    plans + grounds an ordered list of ToolCalls), and score the predicted plan against the gold
    ``episode_plan(ep)`` / ``episode_steps(ep)``. We report four free-run planner metrics plus a
    teacher-forced next-tool metric (from ``multi_turn_eval``) side by side.

    Metrics (returned dict keys):
      whole_plan_acc      — fraction of episodes whose predicted ordered tool-NAME sequence exactly
                            equals the gold plan (same length AND same names in order).
      step_acc            — positional tool-name accuracy. Scored over ``max(len(pred), len(gold))``
                            positions: a position counts correct only if BOTH plans have a step
                            there and the names match. Positions past the shorter plan's end are
                            counted as wrong, so over- AND under-length plans are penalized (an
                            over-length plan can never reach 1.0). Denominator is the sum of
                            ``max(len(pred),len(gold))`` over episodes (empty/empty episodes
                            contribute 0/0 and are skipped).
      grounded_acc        — over the positions where the tool NAME matches, fraction whose grounded
                            ARGS also AST-match (ToolCall.normalized via match_calls). Denominator
                            is the number of name-matched positions.
      plan_len_acc        — fraction of episodes with len(pred) == len(gold).
      by_gold_len         — {gold_len: {"whole": acc, "n": count}} breakdown of whole-plan accuracy
                            bucketed by gold plan length (0..max_steps), to localize failures.
      teacher_forced      — multi_turn_eval(...) on the same episodes: next-tool step_acc /
                            episode_acc when the model is fed the GOLD prior steps (low-variance
                            robustness number alongside the free-run planner metrics).
      episodes, steps     — episode count and total gold steps scored.
    """
    from collections import defaultdict

    from localagent.agent.caller import plan_rollout
    from localagent.data.agent_synth import episode_plan, episode_steps
    from localagent.eval.tool_eval import match_calls

    whole_ok = 0
    len_ok = 0
    step_ok = 0
    step_tot = 0
    grnd_ok = 0
    grnd_tot = 0
    by_len = defaultdict(lambda: [0, 0])   # gold_len -> [whole_ok, n]
    gold_steps_total = 0

    for ep in episodes:
        gold_names = episode_plan(ep)
        gold_calls = episode_steps(ep)
        query = _first_user_query(ep)
        pred_calls = plan_rollout(model, tok, query, tools, tool_head=tool_head,
                                  ptr_head=ptr_head, max_steps=max_steps, device=device)
        pred_names = [c.name for c in pred_calls]

        gold_steps_total += len(gold_names)
        whole = pred_names == gold_names
        whole_ok += whole
        len_ok += (len(pred_names) == len(gold_names))

        # positional step accuracy over the longer of the two (penalizes over/under length)
        span = max(len(pred_names), len(gold_names))
        step_tot += span
        for i in range(span):
            if i < len(pred_names) and i < len(gold_names) and pred_names[i] == gold_names[i]:
                step_ok += 1
                # grounded args only checked where the tool name matched
                grnd_tot += 1
                if match_calls([pred_calls[i]], [gold_calls[i]]):
                    grnd_ok += 1

        b = by_len[len(gold_names)]
        b[0] += whole
        b[1] += 1

    n = len(episodes)
    teacher = multi_turn_eval(model, episodes, tok, tools, device=device,
                              tool_head=tool_head, ptr_head=ptr_head)
    return {
        "whole_plan_acc": whole_ok / max(1, n),
        "step_acc": step_ok / max(1, step_tot),
        "grounded_acc": grnd_ok / max(1, grnd_tot),
        "plan_len_acc": len_ok / max(1, n),
        "by_gold_len": {ln: {"whole": c / max(1, t), "n": t} for ln, (c, t) in sorted(by_len.items())},
        "teacher_forced": {"step_acc": teacher["step_acc"], "episode_acc": teacher["episode_acc"]},
        "episodes": n,
        "steps": gold_steps_total,
    }


def format_plan_eval(res: dict) -> str:
    """One-line-per-section `log`-style summary of plan_eval(), matching analyze_loop's print style."""
    by_len = " ".join(f"L{ln}={v['whole']*100:.0f}%({v['n']})" for ln, v in res["by_gold_len"].items())
    tf = res["teacher_forced"]
    return (
        f"planner: whole={res['whole_plan_acc']*100:.0f}% step={res['step_acc']*100:.0f}% "
        f"grounded={res['grounded_acc']*100:.0f}% plan_len={res['plan_len_acc']*100:.0f}% "
        f"({res['episodes']} eps, {res['steps']} steps)\n"
        f"  by gold-len: {by_len}\n"
        f"  teacher-forced: step_acc={tf['step_acc']*100:.0f}% episode_acc={tf['episode_acc']*100:.0f}%"
    )


def run(checkpoint: str, suite: str = "all", out: str = "runs/eval/report.json") -> dict:
    raise NotImplementedError("Use scripts/flywheel.py — evaluate() is called in-process there")


def parity_check(reference_model, exported_path: str, prompts: list[str]) -> dict:
    raise NotImplementedError("TODO(phase-9): compare exported runtime vs PyTorch reference")
