import copy
import hashlib
import json
import random
from types import SimpleNamespace

import pytest
import torch

from openlocalagent.data.corpus.pretrain_corpus import CorpusDocument, PackedShardDataset, pack_shards
from openlocalagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from openlocalagent.data.schema import Conversation, Message, Role
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer, train_bpe
from openlocalagent.train.midtrain import (
    ConversationDataset,
    ConversationTokenCountDataset,
    MixtureSource,
    ScheduledMixture,
    _audit_packed_holdout_splits,
    assert_checkpoint_compatible,
    midtrain,
    validate_packed_source,
)
from openlocalagent.train.stage_sampling import (
    next_midtrain_microbatch,
    next_midtrain_microbatch_counts,
)


class ToyRows:
    def __init__(self, token):
        self.token = token

    def sample_batch(self, batch_size, rng, device):
        x = torch.full((batch_size, 8), self.token, dtype=torch.long, device=device)
        y = x.clone()
        return x, y


class VariableCountedRows:
    def __init__(
        self,
        token: int,
        loss_lengths: tuple[int, ...],
        *,
        masked_prefix: int = 0,
    ):
        self.token = token
        self.loss_lengths = loss_lengths
        self.masked_prefix = masked_prefix

    def sample_batch_with_counts(self, batch_size, rng, device):
        loss_lengths = [
            self.loss_lengths[rng.randrange(len(self.loss_lengths))] for _ in range(batch_size)
        ]
        input_lengths = [length + self.masked_prefix for length in loss_lengths]
        width = max(input_lengths)
        x = torch.full((batch_size, width), self.token, dtype=torch.long, device=device)
        y = torch.full((batch_size, width), -100, dtype=torch.long, device=device)
        for index, loss_length in enumerate(loss_lengths):
            y[index, self.masked_prefix : self.masked_prefix + loss_length] = self.token
        return x, y, sum(input_lengths), sum(loss_lengths)


def _tiny_model(initial_state=None):
    cfg = ModelConfig(
        vocab_size=256,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=64,
        dropout=0.0,
    )
    model = LocalAgentLM(cfg)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    return model


def _variable_conversation_dataset() -> ConversationDataset:
    conversations = [
        Conversation(
            messages=[
                Message(Role.user, "short prompt"),
                Message(Role.assistant, "ok"),
            ]
        ),
        Conversation(
            messages=[
                Message(Role.user, "a substantially longer prompt " * 2),
                Message(Role.assistant, "a longer supervised answer"),
            ]
        ),
    ]
    return ConversationDataset(conversations, ByteTokenizer(), max_seq_len=64)


@pytest.mark.parametrize(
    "conversation_prompt_contract",
    [None, OPENAI_FULL_CATALOG_V1],
    ids=["legacy", "full-catalog"],
)
def test_count_only_midtrain_sampling_matches_materialized_rows_and_schedule(
    tmp_path,
    conversation_prompt_contract,
):
    if conversation_prompt_contract is None:
        tokenizer = ByteTokenizer()
    else:
        tokenizer = train_bpe(
            [
                (
                    "catalog prompt first answer follow up second answer packed rows "
                    "short and substantially longer deterministic training text"
                )
            ],
            tmp_path / "tokenizer.json",
            vocab_size=300,
            min_frequency=1,
        )
    shards = tmp_path / "shards"
    pack_shards(
        [
            CorpusDocument("short packed row", doc_id="short"),
            CorpusDocument("substantially longer packed training row " * 8, doc_id="long"),
        ],
        tokenizer,
        seq_len=32,
        shards_dir=str(shards),
        rows_per_shard=3,
        val_fraction=0.0,
        seed=7,
    )
    packed = PackedShardDataset(shards, "train")
    conversations = [
        Conversation(
            messages=[
                Message(role=Role.user, content="first prompt"),
                Message(role=Role.assistant, content="first answer"),
                Message(role=Role.user, content="follow up"),
                Message(role=Role.assistant, content="second answer"),
            ]
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="a substantially longer prompt " * 2),
                Message(role=Role.assistant, content="a longer supervised answer"),
            ]
        ),
    ]
    conversation_rows = ConversationDataset(
        conversations,
        tokenizer,
        max_seq_len=4096,
        conversation_prompt_contract=conversation_prompt_contract,
    )
    conversation_counts = ConversationTokenCountDataset(
        conversations,
        tokenizer,
        max_seq_len=4096,
        conversation_prompt_contract=conversation_prompt_contract,
    )
    materialized_count_rng = random.Random(919)
    planner_count_rng = random.Random(919)
    for _ in range(20):
        assert conversation_rows.sample_batch_token_counts(
            3,
            materialized_count_rng,
        ) == conversation_counts.sample_batch_token_counts(3, planner_count_rng)
    assert materialized_count_rng.getstate() == planner_count_rng.getstate()

    def mixture(unit: str, conversation_dataset) -> ScheduledMixture:
        return ScheduledMixture(
            [
                MixtureSource("packed", packed, 0.8, 0.2),
                MixtureSource("conversation", conversation_dataset, 0.2, 0.8),
            ],
            unit=unit,
        )

    for unit in ("draws", "input_tokens", "loss_tokens"):
        materialized_mixture = mixture(unit, conversation_rows)
        count_only_mixture = mixture(unit, conversation_counts)
        materialized_state = materialized_mixture.initial_state()
        count_only_state = count_only_mixture.initial_state()
        materialized_rng = random.Random(41)
        count_only_rng = random.Random(41)
        selected_sources = set()
        for step in range(20):
            for _ in range(2):
                materialized = next_midtrain_microbatch(
                    materialized_mixture,
                    materialized_state,
                    materialized_rng,
                    step=step,
                    total_steps=20,
                    batch_size=3,
                    device="cpu",
                )
                counted = next_midtrain_microbatch_counts(
                    count_only_mixture,
                    count_only_state,
                    count_only_rng,
                    step=step,
                    total_steps=20,
                    batch_size=3,
                )
                selected_sources.add(materialized.source.name)
                assert counted.source.name == materialized.source.name
                assert counted.input_tokens == materialized.input_tokens
                assert counted.loss_tokens == materialized.loss_tokens
                assert count_only_rng.getstate() == materialized_rng.getstate()
                assert count_only_state == materialized_state
        assert selected_sources == {"packed", "conversation"}


def test_scheduled_mixture_weights_interpolate_and_normalize():
    mixture = ScheduledMixture(
        [
            MixtureSource("general", ToyRows(1), 0.8, 0.2),
            MixtureSource("agent", ToyRows(2), 0.2, 0.8),
        ]
    )
    assert mixture.weights_at(0.0) == [0.8, 0.2]
    assert mixture.weights_at(1.0) == pytest.approx([0.2, 0.8])
    assert mixture.weights_at(0.5) == [0.5, 0.5]
    rng1, rng2 = random.Random(4), random.Random(4)
    assert [mixture.choose(0.7, rng1).name for _ in range(20)] == [
        mixture.choose(0.7, rng2).name for _ in range(20)
    ]


def test_scheduled_mixture_interpolates_normalized_endpoint_shares():
    mixture = ScheduledMixture(
        [
            MixtureSource("general", ToyRows(1), 2.0, 1.0),
            MixtureSource("agent", ToyRows(2), 1.0, 1.0),
        ]
    )
    assert mixture.weights_at(0.0) == pytest.approx([2 / 3, 1 / 3])
    assert mixture.weights_at(0.5) == pytest.approx([7 / 12, 5 / 12])
    assert mixture.weights_at(1.0) == pytest.approx([0.5, 0.5])


def test_token_mixture_honors_zero_weight_schedule_endpoints():
    mixture = ScheduledMixture(
        [
            MixtureSource("start_only", ToyRows(1), 1.0, 0.0),
            MixtureSource("end_only", ToyRows(2), 0.0, 1.0),
        ],
        unit="loss_tokens",
    )
    state = mixture.initial_state()
    rng = random.Random(8)

    first = mixture.choose(0.0, rng, state)
    assert first.name == "start_only"
    mixture.observe(
        state,
        first,
        progress=0.0,
        input_tokens=8,
        loss_tokens=8,
    )
    assert mixture.choose(1.0, rng, state).name == "end_only"


def test_loss_token_mixture_tracks_variable_length_conversations_and_reports_shares():
    agent_rows = _variable_conversation_dataset()
    sampled_counts = {
        agent_rows.sample_batch_with_counts(1, random.Random(seed), "cpu")[3] for seed in range(12)
    }
    assert len(sampled_counts) == 2

    mixture = ScheduledMixture(
        [
            MixtureSource(
                "general",
                VariableCountedRows(65, (48,)),
                0.5,
                0.5,
            ),
            MixtureSource("agent", agent_rows, 0.5, 0.5),
        ],
        unit="loss_tokens",
    )
    _, metrics = midtrain(
        _tiny_model(),
        mixture,
        steps=20,
        batch_size=1,
        accum_steps=4,
        lr=1e-3,
        warmup=0,
        device="cpu",
        seed=31,
        return_metrics=True,
        log=lambda *_: None,
    )

    report = metrics["mixture_accounting"]
    assert report["unit"] == "loss_tokens"
    assert report["selection"] == "largest_integrated_token_deficit"
    assert report["loss_normalization"] == "supervised_tokens_across_accumulation"
    assert report["observations"] == 80
    assert report["totals"]["scheduled_target_basis_units"] == pytest.approx(
        report["totals"]["loss_tokens"]
    )
    general = report["sources"]["general"]
    agent = report["sources"]["agent"]
    assert agent["start_weight"] == agent["end_weight"] == 0.5
    assert abs(agent["loss_token_share"] - 0.5) < 0.05
    assert agent["realized_basis_share"] == agent["loss_token_share"]
    assert agent["basis_share_error"] == pytest.approx(
        agent["realized_basis_share"] - agent["scheduled_target_basis_share"]
    )
    assert agent["draw_share"] > agent["loss_token_share"]
    assert general["input_token_share"] + agent["input_token_share"] == pytest.approx(1.0)
    assert general["loss_token_share"] + agent["loss_token_share"] == pytest.approx(1.0)
    assert general["scheduled_target_basis_share"] + agent[
        "scheduled_target_basis_share"
    ] == pytest.approx(1.0)


def test_token_mixture_resume_replays_sampler_and_optimizer_exactly(tmp_path):
    def make_mixture():
        return ScheduledMixture(
            [
                MixtureSource(
                    "general",
                    VariableCountedRows(65, (20, 24)),
                    0.7,
                    0.4,
                ),
                MixtureSource(
                    "agent",
                    VariableCountedRows(66, (2, 4, 6), masked_prefix=5),
                    0.3,
                    0.6,
                ),
            ],
            unit="loss_tokens",
        )

    torch.manual_seed(901)
    initial_state = copy.deepcopy(_tiny_model().state_dict())
    uninterrupted = _tiny_model(initial_state)
    expected_history, expected_metrics = midtrain(
        uninterrupted,
        make_mixture(),
        steps=4,
        batch_size=1,
        accum_steps=2,
        lr=1e-3,
        warmup=0,
        seed=19,
        return_metrics=True,
        log=lambda *_: None,
    )

    checkpoint_path = tmp_path / "midtrain.pt"
    interrupted = _tiny_model(initial_state)
    real_forward = interrupted.forward
    forward_calls = 0

    def crash_during_third_step(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        if forward_calls == 5:
            raise RuntimeError("simulated interruption")
        return real_forward(*args, **kwargs)

    interrupted.forward = crash_during_third_step
    lineage = {"version": 1, "config_sha256": "token-faithful-test"}
    with pytest.raises(RuntimeError, match="simulated interruption"):
        midtrain(
            interrupted,
            make_mixture(),
            steps=4,
            batch_size=1,
            accum_steps=2,
            lr=1e-3,
            warmup=0,
            seed=19,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
            lineage=lineage,
            return_metrics=True,
            log=lambda *_: None,
        )
    periodic = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert periodic["step"] == 1
    assert periodic["mixture_state"]["unit"] == "loss_tokens"

    resumed = _tiny_model(initial_state)
    actual_history, actual_metrics = midtrain(
        resumed,
        make_mixture(),
        steps=4,
        batch_size=1,
        accum_steps=2,
        lr=1e-3,
        warmup=0,
        seed=19,
        checkpoint_path=checkpoint_path,
        resume_from=checkpoint_path,
        lineage=lineage,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert actual_history == pytest.approx(expected_history, rel=0, abs=0)
    assert actual_metrics["source_draws"] == expected_metrics["source_draws"]
    assert actual_metrics["token_accounting"] == expected_metrics["token_accounting"]
    assert actual_metrics["mixture_accounting"] == expected_metrics["mixture_accounting"]
    for name, tensor in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("optimizer", "optimizer state"),
        ("python_rng", "Python RNG state"),
        ("torch_rng", "Torch RNG state"),
        ("global_tokens", "disagrees with per-source sum"),
        ("history", "loss_history length"),
        ("observations", "completed optimizer steps"),
    ],
)
def test_token_mixture_resume_rejects_incomplete_or_inconsistent_state(
    tmp_path,
    mutation,
    message,
):
    checkpoint_path = tmp_path / f"{mutation}.pt"
    mixture = ScheduledMixture(
        [MixtureSource("agent", VariableCountedRows(66, (4,)), 1.0, 1.0)],
        unit="loss_tokens",
    )
    midtrain(
        _tiny_model(),
        mixture,
        steps=1,
        batch_size=1,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        return_metrics=True,
        log=lambda *_: None,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if mutation == "optimizer":
        checkpoint.pop("optimizer")
    elif mutation == "python_rng":
        checkpoint.pop("rng_state")
    elif mutation == "torch_rng":
        checkpoint.pop("torch_rng_state")
    elif mutation == "global_tokens":
        checkpoint["token_accounting"]["loss_tokens"] += 1
    elif mutation == "history":
        checkpoint["loss_history"] = []
    else:
        checkpoint["mixture_state"]["observations"] = 0
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=message):
        midtrain(
            _tiny_model(),
            mixture,
            steps=2,
            batch_size=1,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
            return_metrics=True,
            log=lambda *_: None,
        )


def test_token_mixture_resume_rejects_resolved_execution_drift(tmp_path):
    checkpoint_path = tmp_path / "execution.pt"
    mixture = ScheduledMixture(
        [MixtureSource("agent", VariableCountedRows(66, (4,)), 1.0, 1.0)],
        unit="loss_tokens",
    )
    execution = {"resolved_device": "cpu", "resolved_dtype": "fp32"}
    midtrain(
        _tiny_model(),
        mixture,
        steps=1,
        batch_size=1,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        execution=execution,
        return_metrics=True,
        log=lambda *_: None,
    )

    with pytest.raises(ValueError, match="execution identity mismatch"):
        midtrain(
            _tiny_model(),
            mixture,
            steps=2,
            batch_size=1,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
            execution={"resolved_device": "cpu", "resolved_dtype": "bf16"},
            return_metrics=True,
            log=lambda *_: None,
        )


def test_heldout_evaluation_never_enters_training_mixture_accounting():
    mixture = ScheduledMixture(
        [
            MixtureSource("general", VariableCountedRows(65, (16,)), 0.5, 0.5),
            MixtureSource("agent", VariableCountedRows(66, (4, 8)), 0.5, 0.5),
        ],
        unit="input_tokens",
    )
    heldout = [MixtureSource("heldout_only", VariableCountedRows(67, (12,)), 1.0, 1.0)]
    _, metrics = midtrain(
        _tiny_model(),
        mixture,
        steps=3,
        batch_size=1,
        accum_steps=2,
        lr=1e-3,
        warmup=0,
        seed=44,
        eval_sources=heldout,
        eval_batches=2,
        eval_batch_size=1,
        eval_seed=900,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert set(metrics["mixture_accounting"]["sources"]) == {"general", "agent"}
    assert set(metrics["token_accounting"]["sources"]) == {"general", "agent"}
    assert metrics["mixture_accounting"]["observations"] == 6
    assert metrics["heldout_eval"]["pre"]["sources"]["heldout_only"]["loss_tokens"] == 24
    assert metrics["heldout_eval"]["post"]["sources"]["heldout_only"]["loss_tokens"] == 24


def test_midtrain_runs_and_records_source_draws():
    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=16,
    )
    model = LocalAgentLM(cfg)
    mixture = ScheduledMixture(
        [
            MixtureSource("general", ToyRows(65), 0.5, 0.5),
            MixtureSource("agent", ToyRows(66), 0.5, 0.5),
        ]
    )
    history, draws = midtrain(
        model,
        mixture,
        steps=2,
        batch_size=2,
        lr=1e-3,
        warmup=1,
        device="cpu",
        log=lambda *_: None,
    )
    assert len(history) == 2
    assert sum(draws.values()) == 2
    assert all(torch.isfinite(torch.tensor(history)))


def test_checkpoint_compatibility_rejects_same_shape_rope_change():
    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=16,
    )
    checkpoint_cfg = dict(cfg.__dict__)
    checkpoint_cfg["rope_theta"] = cfg.rope_theta * 2

    with pytest.raises(ValueError, match="rope_theta"):
        assert_checkpoint_compatible({"cfg": checkpoint_cfg}, cfg)


def test_checkpoint_compatibility_accepts_resolved_all_attention_equivalence():
    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=16,
        layer_types=["attn"],
    )
    checkpoint_cfg = {**cfg.__dict__, "layer_types": None}

    assert_checkpoint_compatible({"cfg": checkpoint_cfg}, cfg)


def test_packed_source_validates_vocab_and_tokenizer_fingerprint():
    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=16,
    )
    dataset = SimpleNamespace(
        seq_len=16,
        manifest={
            "vocab_size": 256,
            "tokenizer_training": {"artifact": {"sha256": "tokenizer-a"}},
        },
    )

    assert (
        validate_packed_source(
            dataset,
            cfg,
            source_name="general",
            configured_tokenizer_sha256="tokenizer-a",
        )
        == "tokenizer-a"
    )
    with pytest.raises(ValueError, match="fingerprint"):
        validate_packed_source(
            dataset,
            cfg,
            source_name="general",
            configured_tokenizer_sha256="tokenizer-b",
        )

    dataset.manifest["vocab_size"] = 320
    with pytest.raises(ValueError, match="vocabulary"):
        validate_packed_source(
            dataset,
            cfg,
            source_name="general",
            configured_tokenizer_sha256="tokenizer-a",
        )


def test_packed_holdout_audit_proves_document_disjoint_splits(tmp_path):
    docs = [
        CorpusDocument(
            f"Auditable packed document {index}. " * 20,
            doc_id=f"document-{index}",
        )
        for index in range(12)
    ]
    pack_shards(
        docs,
        ByteTokenizer(),
        seq_len=16,
        shards_dir=str(tmp_path),
        rows_per_shard=8,
        val_fraction=0.25,
        seed=19,
    )
    train = PackedShardDataset(tmp_path, "train")
    heldout = PackedShardDataset(tmp_path, "val")

    audit = _audit_packed_holdout_splits(
        [("train", train)],
        [("heldout", heldout)],
    )

    assert audit is not None
    assert audit["proof"] == "verified_content_bound_split_assignment_rows"
    assert audit["pairs"] == [
        {
            "train_source": "train",
            "eval_source": "heldout",
            "document_identity_overlap": 0,
            "document_content_overlap": 0,
        }
    ]
    assert audit["train"][0]["documents"] + audit["eval"][0]["documents"] == 12


def test_packed_holdout_audit_accepts_repeated_id_with_distinct_content_bindings(
    tmp_path,
):
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    pack_shards(
        [
            CorpusDocument("First version of the repeated identity. " * 20, doc_id="repeated"),
            CorpusDocument("Second version of the repeated identity. " * 20, doc_id="repeated"),
        ],
        ByteTokenizer(),
        seq_len=16,
        shards_dir=str(train_dir),
        rows_per_shard=8,
        val_fraction=0.0,
        seed=11,
    )
    pack_shards(
        [
            CorpusDocument("Independent held-out document. " * 20, doc_id="heldout"),
        ],
        ByteTokenizer(),
        seq_len=16,
        shards_dir=str(eval_dir),
        rows_per_shard=8,
        val_fraction=0.0,
        seed=11,
    )

    audit = _audit_packed_holdout_splits(
        [("train", PackedShardDataset(train_dir, "train"))],
        [("heldout", PackedShardDataset(eval_dir, "train"))],
    )

    assert audit is not None
    assert audit["train"][0]["documents"] == 2
    assert audit["train"][0]["unique_document_identities"] == 1
    assert audit["pairs"][0]["document_identity_overlap"] == 0
    assert audit["pairs"][0]["document_content_overlap"] == 0


def test_packed_holdout_audit_rejects_one_identity_bound_across_splits(tmp_path):
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    pack_shards(
        [
            CorpusDocument("First repeated binding. " * 20, doc_id="repeated"),
            CorpusDocument("Second repeated binding. " * 20, doc_id="repeated"),
        ],
        ByteTokenizer(),
        seq_len=16,
        shards_dir=str(train_dir),
        rows_per_shard=8,
        val_fraction=0.0,
        seed=13,
    )
    pack_shards(
        [CorpusDocument("Clean held-out binding. " * 20, doc_id="heldout")],
        ByteTokenizer(),
        seq_len=16,
        shards_dir=str(eval_dir),
        rows_per_shard=8,
        val_fraction=0.0,
        seed=13,
    )
    manifest_path = train_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignment_path = train_dir / manifest["split_assignment"]["path"]
    rows = [json.loads(line) for line in assignment_path.read_text(encoding="utf-8").splitlines()]
    rows[2]["split"] = "val"
    payload = "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    assignment_path.write_bytes(payload)
    assignment_values = [
        f"{row['identity_sha256']}:{row['document_sha256']}:{row['split']}" for row in rows[1:]
    ]
    assignment_sha256 = hashlib.sha256("\n".join(assignment_values).encode("ascii")).hexdigest()
    manifest["split_assignment"]["bytes"] = len(payload)
    manifest["split_assignment"]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["split_assignment"]["assignment_sha256"] = assignment_sha256
    manifest["split_assignment_sha256"] = assignment_sha256
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="assigned to multiple splits"):
        _audit_packed_holdout_splits(
            [("train", PackedShardDataset(train_dir, "train"))],
            [("heldout", PackedShardDataset(eval_dir, "train"))],
        )


def test_packed_holdout_audit_rejects_overlap_across_distinct_artifacts(tmp_path):
    shared = CorpusDocument("Shared held-out contamination. " * 20, doc_id="shared")
    for name, unique in (
        ("train", CorpusDocument("Training-only content. " * 20, doc_id="train-only")),
        ("eval", CorpusDocument("Evaluation-only content. " * 20, doc_id="eval-only")),
    ):
        pack_shards(
            [shared, unique],
            ByteTokenizer(),
            seq_len=16,
            shards_dir=str(tmp_path / name),
            rows_per_shard=8,
            val_fraction=0.0,
            seed=3,
        )

    with pytest.raises(ValueError, match="packed held-out contamination"):
        _audit_packed_holdout_splits(
            [("train", PackedShardDataset(tmp_path / "train", "train"))],
            [("heldout", PackedShardDataset(tmp_path / "eval", "train"))],
        )


def test_packed_holdout_audit_fails_closed_without_assignment_proof(tmp_path):
    docs = [
        CorpusDocument(f"Proof-required document {index}. " * 20, doc_id=f"proof-{index}")
        for index in range(4)
    ]
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    for directory in (train_dir, eval_dir):
        pack_shards(
            docs,
            ByteTokenizer(),
            seq_len=16,
            shards_dir=str(directory),
            rows_per_shard=8,
            val_fraction=0.0,
            seed=5,
        )
    eval_manifest_path = eval_dir / "manifest.json"
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    eval_manifest.pop("split_assignment")
    eval_manifest.pop("split_assignment_sha256")
    eval_manifest_path.write_text(
        json.dumps(eval_manifest, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot prove packed held-out disjointness"):
        _audit_packed_holdout_splits(
            [("train", PackedShardDataset(train_dir, "train"))],
            [("heldout", PackedShardDataset(eval_dir, "train"))],
        )
