"""Fetch a small public text sample for toy pretraining (Phase 2)."""

from __future__ import annotations

from localagent.data.pretrain_corpus import download_sample

if __name__ == "__main__":
    download_sample("data/raw")
