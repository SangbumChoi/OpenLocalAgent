import hashlib
import json
from argparse import Namespace

from scripts.run_gpu_campaign import run_campaign


def test_gpu_campaign_records_matrix_and_model_inventory_without_training(tmp_path) -> None:
    args = Namespace(
        campaign="configs/experiments/webgpu-realistic-campaign.v1.yaml",
        output=tmp_path,
        models=["ultra-tiny-1m"],
        device="cpu",
        dtype="fp32",
        prompt_len=4,
        decode=2,
        repeats=1,
        no_uncached=True,
        skip_benchmark=True,
        skip_matrix=False,
        matrix="configs/data/realistic-agent-public-eval-matrix.v1.json",
        transfer=None,
        checkpoint=None,
        acquire_hf=False,
        hf_config="configs/experiments/hf-sources.v1.yaml",
        hf_out=tmp_path / "hf",
        hf_sources=None,
        hf_dry_run=False,
        wandb=False,
        wandb_project="localagent",
        wandb_entity=None,
        wandb_run_name=None,
        wandb_mode="offline",
        force=False,
    )
    report = run_campaign(args)
    assert report["kind"] == "localagent_gpu_realistic_campaign"
    assert report["model_inventory"][0]["name"] == "ultra-tiny-1m"
    assert report["model_inventory"][0]["budget_pass"] is True
    assert report["public_matrix"]["entries"] == 28
    assert report["wandb"] == {"enabled": False}
    body = {key: value for key, value in report.items() if key != "receipt_self_sha256"}
    assert report["receipt_self_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    saved = json.loads((tmp_path / "campaign.json").read_text(encoding="utf-8"))
    assert saved["receipt_self_sha256"] == report["receipt_self_sha256"]
