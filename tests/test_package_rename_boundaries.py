"""The package is `openlocalagent`; the on-disk formats it writes are still `localagent_*`.

Renaming the package is a source change. Renaming a format tag is a data migration, because the
tag is written into every manifest, receipt and shard the project has produced - `pt-big` and
`pt-4b` both carry `localagent_document_split_jsonl` right now. A blanket rename broke four tests
by moving those tags out from under the files that already contain them.

The hash personalization is the sharper case: blake2b caps it at 16 bytes, so
b"openlocalagent-data" does not merely change the hash, it fails outright - and had it fit, it
would have silently changed which documents near-dedup considers duplicates.
"""

import re
from pathlib import Path

import pytest

SOURCE = sorted(Path("src/openlocalagent").rglob("*.py"))
#: Dotted or slashed - a module path. Underscored - a value written to disk.
MODULE_PATH = re.compile(r"(?<!open)localagent[./]")


def test_the_package_directory_is_the_new_name():
    assert Path("src/openlocalagent").is_dir()
    assert not Path("src/localagent").exists()


@pytest.mark.parametrize("path", SOURCE, ids=lambda p: str(p))
def test_no_source_file_imports_the_old_module_path(path: Path):
    body = path.read_text(encoding="utf-8")
    stray = MODULE_PATH.findall(body)
    assert not stray, f"{path} still refers to the old module path: {set(stray)}"


def test_format_tags_kept_their_original_spelling():
    """A tag identifies a file format. The format did not change, so neither did its name."""
    from openlocalagent.data.corpus import pretrain_corpus

    tags = set()
    for path in SOURCE:
        tags |= set(re.findall(r"\blocalagent_[a-z_]+", path.read_text(encoding="utf-8")))
    assert tags, "no on-disk format tags found; this guard would be vacuous"
    assert not any(tag.startswith("openlocalagent_") for tag in tags)
    # The one that fails loudly rather than silently.
    assert pretrain_corpus._SHINGLE_HASH_PERSON == b"localagent-data"


def test_hash_personalization_still_fits_blake2b():
    """16 bytes is the hard limit; "openlocalagent-data" is 19 and raises at hash time."""
    import hashlib

    from openlocalagent.data.corpus.pretrain_corpus import _SHINGLE_HASH_PERSON

    assert len(_SHINGLE_HASH_PERSON) <= 16
    hashlib.blake2b(b"x", digest_size=8, person=_SHINGLE_HASH_PERSON)
    with pytest.raises(ValueError):
        hashlib.blake2b(b"x", digest_size=8, person=b"openlocalagent-data")


def test_console_scripts_use_the_new_name():
    body = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "openlocalagent"' in body
    assert "openlocalagent = \"openlocalagent.cli:main\"" in body
    assert "openlocalagent-eval-suite" in body
