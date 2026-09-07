import copy

import pytest
import torch
import yaml

from openlocalagent.data.corpus.pretrain_corpus import pack_shards
from openlocalagent.model import LocalAgentLM, ModelConfig
from openlocalagent.model.tokenizer import ByteTokenizer
from openlocalagent.train.pretrain import pretrain
from openlocalagent.train.pretrain import run as run_pretrain


def _model():
    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
    )
    return LocalAgentLM(cfg)


def _hybrid_model():
    cfg = ModelConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=32,
        layer_types=["conv", "attn"],
    )
    return LocalAgentLM(cfg)


def test_pretrain_checkpoint_resume_continues_optimizer_steps(tmp_path):
    stream = list(("agent tools and code " * 100).encode())
    checkpoint = tmp_path / "latest.pt"
    first = pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=2,
        batch_size=2,
        seq_len=16,
        lr=1e-3,
        warmup=1,
        checkpoint_path=checkpoint,
        device="cpu",
        log=lambda *_: None,
    )
    assert len(first) == 2
    second = pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=4,
        batch_size=2,
        seq_len=16,
        lr=1e-3,
        warmup=1,
        checkpoint_path=checkpoint,
        resume_from=checkpoint,
        device="cpu",
        log=lambda *_: None,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert len(second) == 4
    assert saved["step"] == 3
    assert saved["tokens_seen"] == 4 * 2 * 16
    assert saved["training_seed"] == 0


def test_pretrain_resume_rejects_different_training_seed(tmp_path):
    stream = list(("agent tools and code " * 100).encode())
    checkpoint = tmp_path / "latest.pt"
    pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        seq_len=8,
        seed=17,
        checkpoint_path=checkpoint,
        device="cpu",
        log=lambda *_: None,
    )

    with pytest.raises(ValueError, match="training seed mismatch"):
        pretrain(
            _model(),
            stream,
            ByteTokenizer(),
            steps=2,
            batch_size=1,
            seq_len=8,
            seed=18,
            checkpoint_path=checkpoint,
            resume_from=checkpoint,
            device="cpu",
            log=lambda *_: None,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "optimizer",
        "step",
        "training_seed",
        "rng_state",
        "torch_rng_state",
        "grad_scaler",
    ],
)
def test_pretrain_exact_resume_rejects_incomplete_state(tmp_path, missing_field):
    stream = list(("agent tools and code " * 100).encode())
    checkpoint = tmp_path / "latest.pt"
    pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        seq_len=8,
        checkpoint_path=checkpoint,
        device="cpu",
        log=lambda *_: None,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del payload[missing_field]
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="exact-resume checkpoint is incomplete"):
        pretrain(
            _model(),
            stream,
            ByteTokenizer(),
            steps=2,
            batch_size=1,
            seq_len=8,
            checkpoint_path=checkpoint,
            resume_from=checkpoint,
            device="cpu",
            log=lambda *_: None,
        )


class _DeterministicPackedRows:
    def __init__(self, *, interrupt_after: int | None = None):
        self.calls = 0
        self.interrupt_after = interrupt_after

    def sample_batch(self, batch_size, rng, device):
        self.calls += 1
        offset = rng.randint(0, 200)
        if self.interrupt_after is not None and self.calls > self.interrupt_after:
            raise RuntimeError("simulated interruption")
        row = (torch.arange(9, dtype=torch.long) + offset) % 256
        batch = row.repeat(batch_size, 1).to(device)
        return batch[:, :-1], batch[:, 1:]


def test_pretrain_exact_resume_matches_uninterrupted_training(tmp_path):
    seed = 73
    torch.manual_seed(101)
    uninterrupted_model = _model()
    pretrain(
        uninterrupted_model,
        _DeterministicPackedRows(),
        ByteTokenizer(),
        steps=4,
        batch_size=2,
        seq_len=8,
        lr=1e-3,
        warmup=1,
        seed=seed,
        device="cpu",
        log=lambda *_: None,
    )

    checkpoint = tmp_path / "interrupted.pt"
    torch.manual_seed(101)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pretrain(
            _model(),
            _DeterministicPackedRows(interrupt_after=2),
            ByteTokenizer(),
            steps=4,
            batch_size=2,
            seq_len=8,
            lr=1e-3,
            warmup=1,
            seed=seed,
            checkpoint_path=checkpoint,
            checkpoint_every=2,
            device="cpu",
            log=lambda *_: None,
        )
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["step"] == 1

    resumed_model = _model()
    resumed_history = pretrain(
        resumed_model,
        _DeterministicPackedRows(),
        ByteTokenizer(),
        steps=4,
        batch_size=2,
        seq_len=8,
        lr=1e-3,
        warmup=1,
        seed=seed,
        checkpoint_path=checkpoint,
        resume_from=checkpoint,
        device="cpu",
        log=lambda *_: None,
    )

    assert len(resumed_history) == 4
    for name, tensor in uninterrupted_model.state_dict().items():
        assert torch.equal(tensor, resumed_model.state_dict()[name]), name


class _MPSRandomPackedRows:
    def __init__(self, *, interrupt_after: int | None = None):
        self.calls = 0
        self.interrupt_after = interrupt_after

    def sample_batch(self, batch_size, rng, device):
        del rng
        self.calls += 1
        offset = int(torch.randint(0, 201, (1,), device=device).item())
        if self.interrupt_after is not None and self.calls > self.interrupt_after:
            raise RuntimeError("simulated interruption")
        row = (torch.arange(9, dtype=torch.long) + offset) % 256
        batch = row.repeat(batch_size, 1).to(device)
        return batch[:, :-1], batch[:, 1:]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS exact-resume regression requires an available MPS device",
)
def test_pretrain_mps_exact_resume_restores_backend_rng_and_optimizer(tmp_path):
    seed = 73
    torch.manual_seed(101)
    initial_state = copy.deepcopy(_hybrid_model().state_dict())
    execution = {
        "resolved_device": "mps",
        "resolved_dtype": "fp32",
        "torch_version": str(torch.__version__),
    }

    uninterrupted_model = _hybrid_model()
    uninterrupted_model.load_state_dict(initial_state)
    uninterrupted_history = pretrain(
        uninterrupted_model,
        _MPSRandomPackedRows(),
        ByteTokenizer(),
        steps=4,
        batch_size=2,
        seq_len=8,
        lr=1e-3,
        warmup=1,
        seed=seed,
        device="mps",
        execution=execution,
        log=lambda *_: None,
    )

    checkpoint = tmp_path / "interrupted-mps.pt"
    interrupted_model = _hybrid_model()
    interrupted_model.load_state_dict(initial_state)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pretrain(
            interrupted_model,
            _MPSRandomPackedRows(interrupt_after=2),
            ByteTokenizer(),
            steps=4,
            batch_size=2,
            seq_len=8,
            lr=1e-3,
            warmup=1,
            seed=seed,
            checkpoint_path=checkpoint,
            checkpoint_every=2,
            device="mps",
            execution=execution,
            log=lambda *_: None,
        )
    periodic = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert periodic["step"] == 1
    assert periodic["mps_rng_state"].dtype == torch.uint8
    assert periodic["mps_rng_state"].ndim == 1
    assert periodic["cuda_rng_state_all"] is None

    resumed_model = _hybrid_model()
    resumed_model.load_state_dict(initial_state)
    resumed_history = pretrain(
        resumed_model,
        _MPSRandomPackedRows(),
        ByteTokenizer(),
        steps=4,
        batch_size=2,
        seq_len=8,
        lr=1e-3,
        warmup=1,
        seed=seed,
        checkpoint_path=checkpoint,
        resume_from=checkpoint,
        device="mps",
        execution=execution,
        log=lambda *_: None,
    )
    torch.mps.synchronize()

    assert resumed_history == uninterrupted_history
    for name, tensor in uninterrupted_model.state_dict().items():
        assert torch.equal(tensor.cpu(), resumed_model.state_dict()[name].cpu()), name


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS exact-resume validation requires an available MPS device",
)
def test_pretrain_mps_exact_resume_rejects_missing_backend_rng_state(tmp_path):
    stream = list(("agent tools and code " * 100).encode())
    checkpoint = tmp_path / "missing-mps-rng.pt"
    pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        seq_len=8,
        checkpoint_path=checkpoint,
        device="mps",
        log=lambda *_: None,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del payload["mps_rng_state"]
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="MPS RNG state is missing or invalid"):
        pretrain(
            _model(),
            stream,
            ByteTokenizer(),
            steps=2,
            batch_size=1,
            seq_len=8,
            checkpoint_path=checkpoint,
            resume_from=checkpoint,
            device="mps",
            log=lambda *_: None,
        )


def test_pretrain_exact_resume_rejects_resolved_execution_drift(tmp_path):
    stream = list(("agent tools and code " * 100).encode())
    checkpoint = tmp_path / "execution.pt"
    execution = {"resolved_device": "cpu", "resolved_dtype": "fp32"}
    pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=1,
        batch_size=1,
        seq_len=8,
        checkpoint_path=checkpoint,
        device="cpu",
        execution=execution,
        log=lambda *_: None,
    )

    with pytest.raises(ValueError, match="execution identity mismatch"):
        pretrain(
            _model(),
            stream,
            ByteTokenizer(),
            steps=2,
            batch_size=1,
            seq_len=8,
            checkpoint_path=checkpoint,
            resume_from=checkpoint,
            device="cpu",
            execution={"resolved_device": "cpu", "resolved_dtype": "bf16"},
            log=lambda *_: None,
        )


def test_pretrain_atomically_mirrors_checkpoint(tmp_path):
    stream = list(("agent tools and code " * 100).encode())
    checkpoint = tmp_path / "local" / "latest.pt"
    mirror = tmp_path / "drive" / "latest.pt"
    pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=1,
        batch_size=2,
        seq_len=16,
        checkpoint_path=checkpoint,
        checkpoint_mirror_path=mirror,
        device="cpu",
        log=lambda *_: None,
    )

    assert mirror.read_bytes() == checkpoint.read_bytes()
    assert not mirror.with_suffix(".pt.tmp").exists()


class _CountingValidationRows:
    def __init__(self):
        self.calls = 0

    def sample_batch(self, batch_size, rng, device):
        self.calls += 1
        x = torch.full((batch_size, 8), 65, dtype=torch.long, device=device)
        return x, x.clone()


def test_pretrain_eval_every_is_independent_of_coarse_logging():
    stream = list(("agent tools and code " * 100).encode())
    validation = _CountingValidationRows()

    _, metrics = pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=16,
        batch_size=1,
        seq_len=8,
        lr=1e-3,
        warmup=1,
        val_data=validation,
        eval_every=3,
        eval_batches=1,
        device="cpu",
        log=lambda *_: None,
        return_metrics=True,
    )

    assert validation.calls == 6  # steps 0, 3, 6, 9, 12, and 15
    assert [record["step"] for record in metrics["validation_history"]] == [
        0,
        3,
        6,
        9,
        12,
        15,
    ]
    assert metrics["validation_last"] == metrics["validation_history"][-1]


def test_pretrain_checkpoint_preserves_validation_history_across_resume(tmp_path):
    stream = list(("agent tools and code " * 100).encode())
    validation = _CountingValidationRows()
    checkpoint = tmp_path / "latest.pt"

    _, first_metrics = pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=3,
        batch_size=1,
        seq_len=8,
        val_data=validation,
        eval_every=2,
        eval_batches=1,
        checkpoint_path=checkpoint,
        log=lambda *_: None,
        return_metrics=True,
    )
    _, resumed_metrics = pretrain(
        _model(),
        stream,
        ByteTokenizer(),
        steps=5,
        batch_size=1,
        seq_len=8,
        val_data=validation,
        eval_every=2,
        eval_batches=1,
        checkpoint_path=checkpoint,
        resume_from=checkpoint,
        log=lambda *_: None,
        return_metrics=True,
    )

    assert resumed_metrics["validation_history"][:2] == first_metrics["validation_history"]
    assert [record["step"] for record in resumed_metrics["validation_history"]] == [0, 2, 4]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["validation_history"] == resumed_metrics["validation_history"]


def test_pretrain_runner_seed_reproduces_initialization_and_update(tmp_path):
    cfg = ModelConfig(
        name="seeded-pretrain",
        vocab_size=256,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=8,
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(cfg.__dict__), encoding="utf-8")
    shards = tmp_path / "shards"
    pack_shards(
        ["agent tools and deterministic training " * 8],
        ByteTokenizer(),
        seq_len=8,
        shards_dir=str(shards),
        rows_per_shard=8,
        val_fraction=0.0,
    )

    checkpoints = []
    for run_index in range(2):
        out_dir = tmp_path / f"run-{run_index}"
        config_path = tmp_path / f"pretrain-{run_index}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "stage": "pretrain",
                    "model_config": str(model_path),
                    "data": {"shards_dir": str(shards), "tokenizer": {"kind": "byte"}},
                    "optim": {"lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0},
                    "schedule": {
                        "type": "cosine",
                        "warmup_steps": 0,
                        "total_steps": 1,
                    },
                    "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
                    "runtime": {"device": "cpu", "dtype": "fp32", "seed": 91},
                    "log": {"out_dir": str(out_dir)},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        run_pretrain(str(config_path))
        checkpoints.append(
            torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
        )

    assert [checkpoint["training_seed"] for checkpoint in checkpoints] == [91, 91]
    assert [checkpoint["execution"]["resolved_device"] for checkpoint in checkpoints] == [
        "cpu",
        "cpu",
    ]
    assert [checkpoint["execution"]["resolved_dtype"] for checkpoint in checkpoints] == [
        "fp32",
        "fp32",
    ]
    for key in checkpoints[0]["state_dict"]:
        assert torch.equal(
            checkpoints[0]["state_dict"][key],
            checkpoints[1]["state_dict"][key],
        ), key


def test_pretrain_runner_enforces_minimum_unique_train_corpus_tokens(tmp_path):
    cfg = ModelConfig(
        name="corpus-budget-gate",
        vocab_size=256,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=8,
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(cfg.__dict__), encoding="utf-8")
    shards = tmp_path / "shards"
    pack_shards(
        ["small but valid training document " * 4],
        ByteTokenizer(),
        seq_len=8,
        shards_dir=str(shards),
        rows_per_shard=8,
        val_fraction=0.0,
    )
    manifest = yaml.safe_load((shards / "manifest.json").read_text(encoding="utf-8"))
    available = int(manifest["splits"]["train"]["tokens"])
    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "stage": "pretrain",
                "model_config": str(model_path),
                "data": {
                    "shards_dir": str(shards),
                    "min_train_tokens": available + 1,
                    "tokenizer": {"kind": "byte"},
                },
                "schedule": {"total_steps": 0},
                "runtime": {"device": "cpu", "dtype": "fp32", "seed": 7},
                "log": {"out_dir": str(tmp_path / "run")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"available=.*required=",
    ):
        run_pretrain(str(config_path))
