# Online streaming pretraining

`scripts/train_stream_pretrain.py` trains directly from the pinned public sources in
`configs/data/pretrain-paper.yaml`. It does not require a pre-existing JSONL corpus. Each source
has an adapter, and the trainer records per-source loss alongside the overall training curve.

Install the project dependencies, then run a bounded trial:

```bash
python scripts/train_stream_pretrain.py --config configs/pretrain/la-1m-online-100step.yaml
```

The broader source mix uses `configs/pretrain/la-1m-online-all.yaml`. The runner writes checkpoints,
curves, and a summary beneath `results/runs/`, which is ignored by Git. An optional `.env` file may
provide `HF_TOKEN` and `WANDB_API_KEY`; copy `.env.example` and set only the values you need. Never
commit the populated `.env` file. Network access and sufficient cache/disk space are required.

Source revisions, licenses, and mixture weights are pinned in the data config. The run's reported
training loss is not a held-out agent benchmark or evidence of tool-use generalization.
