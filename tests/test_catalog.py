from openlocalagent.agent.retriever import ToolRetriever, embed
from openlocalagent.data.tool_catalog import (VALUES_EVAL, VALUES_TRAIN, build_catalog, gen_episodes,
                                          gen_usages)


def test_build_catalog_size_and_unique():
    tools = build_catalog(300)
    assert len(tools) == 300
    assert len({t.name for t in tools}) == 300  # unique tool names


def test_train_eval_values_disjoint():
    assert not (set(VALUES_TRAIN) & set(VALUES_EVAL))


def test_embed_is_normalized():
    import numpy as np
    v = embed("book a flight")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_example_aug_retrieval_beats_desc_only():
    tools = build_catalog(200, seed=0)
    ex = {}
    for u in gen_usages(tools, "train", per_tool=4, seed=1, paraphrase=True):
        ex.setdefault(u["tool"], []).append(u["prompt"])
    desc, aug = ToolRetriever(tools), ToolRetriever(tools, examples=ex)
    test = gen_usages(tools, "eval", per_tool=1, seed=7)  # paraphrased
    rd = sum(t["tool"] == desc.retrieve(t["prompt"], 1)[0] for t in test) / len(test)
    ra = sum(t["tool"] == aug.retrieve(t["prompt"], 1)[0] for t in test) / len(test)
    assert ra > rd  # indexing by example usages bridges the paraphrase gap


def test_episodes_two_steps():
    eps = gen_episodes(build_catalog(50), n=5, split="eval", seed=0)
    assert all(len(e) == 2 for e in eps)
    assert eps[0][1]["history"]  # step 2 carries history
