import hashlib
from pathlib import Path

from scripts.normalize_xlam_parquet import (
    DERIVED_DATASET,
    DERIVED_REVISION,
    DERIVED_URL,
    _identity,
    _source,
)


def test_xlam_derivative_source_identity_is_hash_bound(tmp_path: Path) -> None:
    source_path = tmp_path / "shard_0.parquet"
    source_path.write_bytes(b"parquet-placeholder")
    identity = _identity(source_path)
    assert identity["bytes"] == len(b"parquet-placeholder")
    assert identity["sha256"] == hashlib.sha256(b"parquet-placeholder").hexdigest()
    source = _source(source_path, split="train", identity=identity)
    assert source.dataset == DERIVED_DATASET
    assert source.revision == DERIVED_REVISION
    assert source.split == "train"
    assert source.license == "apache-2.0"


def test_xlam_derivative_claim_boundary_is_not_official() -> None:
    # Keep this assertion close to the adapter's public contract: derivative records can be
    # useful for training, but must never silently become an official Salesforce benchmark claim.
    assert "product-science" in DERIVED_DATASET
    assert DERIVED_URL.endswith("xlam-function-calling-60k-raw")
