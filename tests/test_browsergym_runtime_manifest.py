from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from openlocalagent.data.adapters import browsergym_runtime_manifest
from openlocalagent.data.adapters.browsergym_runtime_manifest import (
    BROWSERGYM_RUNTIME_MANIFEST_KIND,
    BROWSERGYM_RUNTIME_MANIFEST_SCHEMA_VERSION,
    build_active_environment_manifest,
    freeze_active_environment_manifest,
    load_and_verify_environment_manifest,
)

_ORIGINAL_SYSCONFIG_PATHS = browsergym_runtime_manifest.sysconfig.get_paths()
_ORIGINAL_SOABI = browsergym_runtime_manifest.sysconfig.get_config_var("SOABI")


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


@dataclass(frozen=True)
class _FakeEntryPoint:
    group: str
    name: str


@dataclass
class _FakeDistribution:
    name: str
    version: str
    declared_locations: dict[str, Path]
    entry_points: tuple[_FakeEntryPoint, ...] = ()
    direct_url: str | None = None
    metadata: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.metadata = {"Name": self.name}

    @property
    def files(self) -> tuple[PurePosixPath, ...]:
        return tuple(PurePosixPath(path) for path in self.declared_locations)

    def locate_file(self, path: object) -> Path:
        return self.declared_locations[str(path)]

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json":
            return self.direct_url
        return None


@dataclass
class _SyntheticEnvironment:
    root: Path
    purelib: Path
    scripts: Path
    executable: Path
    distributions: list[_FakeDistribution]
    demo_module: Path
    demo_distribution: _FakeDistribution
    playwright_distribution: _FakeDistribution
    driver: Path

    def activate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            browsergym_runtime_manifest.sysconfig,
            "get_paths",
            lambda *args, **kwargs: {
                **_ORIGINAL_SYSCONFIG_PATHS,
                "purelib": str(self.purelib),
                "platlib": str(self.purelib),
                "scripts": str(self.scripts),
                "data": str(self.root),
            },
        )
        monkeypatch.setattr(
            browsergym_runtime_manifest.sysconfig,
            "get_config_var",
            lambda name: _ORIGINAL_SOABI if name == "SOABI" else None,
        )
        monkeypatch.setattr(
            browsergym_runtime_manifest.importlib.metadata,
            "distributions",
            lambda **kwargs: tuple(self.distributions),
        )
        monkeypatch.setattr(
            browsergym_runtime_manifest.sys,
            "executable",
            str(self.executable),
        )


def _write_file(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)


def _synthetic_environment(root: Path) -> _SyntheticEnvironment:
    purelib = root / "lib" / "python" / "site-packages"
    scripts = root / "bin"
    purelib.mkdir(parents=True)
    scripts.mkdir(parents=True)
    executable = root / "python"
    _write_file(executable, b"synthetic-python-executable\n", mode=0o755)

    demo_init = purelib / "demo_pkg" / "__init__.py"
    demo_module = purelib / "demo_pkg" / "module.py"
    demo_metadata = purelib / "demo_pkg-1.2.3.dist-info" / "METADATA"
    demo_entry_points = purelib / "demo_pkg-1.2.3.dist-info" / "entry_points.txt"
    demo_direct_url = purelib / "demo_pkg-1.2.3.dist-info" / "direct_url.json"
    demo_record = purelib / "demo_pkg-1.2.3.dist-info" / "RECORD"
    demo_launcher = scripts / "demo-tool"
    _write_file(demo_init, b"")
    _write_file(demo_module, b"VALUE = 1\n")
    _write_file(demo_metadata, b"Name: Demo_Pkg\nVersion: 1.2.3\n")
    _write_file(
        demo_entry_points,
        b"[console_scripts]\ndemo-tool=demo_pkg.module:main\n",
    )
    archive_sha256 = "a" * 64
    direct_url_payload = json.dumps(
        {
            "archive_info": {
                "hash": f"sha256={archive_sha256}",
                "hashes": {"sha256": archive_sha256},
            },
            "url": f"file://{root}/wheels/demo_pkg-1.2.3-py3-none-any.whl",
        },
        sort_keys=True,
    ).encode()
    _write_file(demo_direct_url, direct_url_payload)
    direct_url_record_hash = base64.urlsafe_b64encode(
        hashlib.sha256(direct_url_payload).digest()
    ).rstrip(b"=")
    _write_file(
        demo_record,
        (
            b"demo_pkg-1.2.3.dist-info/RECORD,,\n"
            b"demo_pkg-1.2.3.dist-info/direct_url.json,sha256="
            + direct_url_record_hash
            + f",{len(direct_url_payload)}\n".encode()
        ),
    )
    _write_file(
        demo_launcher,
        f"#!{root}/python\nprint('demo')\n".encode(),
        mode=0o755,
    )
    demo_distribution = _FakeDistribution(
        "Demo_Pkg",
        "1.2.3",
        {
            "demo_pkg/__init__.py": demo_init,
            "demo_pkg/module.py": demo_module,
            "demo_pkg-1.2.3.dist-info/METADATA": demo_metadata,
            "demo_pkg-1.2.3.dist-info/entry_points.txt": demo_entry_points,
            "demo_pkg-1.2.3.dist-info/direct_url.json": demo_direct_url,
            "demo_pkg-1.2.3.dist-info/RECORD": demo_record,
            "../../../bin/demo-tool": demo_launcher,
        },
        entry_points=(_FakeEntryPoint("console_scripts", "demo-tool"),),
        direct_url=direct_url_payload.decode(),
    )

    playwright_init = purelib / "playwright" / "__init__.py"
    driver = purelib / "playwright" / "driver"
    driver_node = driver / "node"
    driver_cli = driver / "package" / "cli.js"
    ignored_bytecode = driver / "__pycache__" / "ignored.pyc"
    playwright_metadata = purelib / "playwright-1.58.0.dist-info" / "METADATA"
    _write_file(playwright_init, b"__version__ = '1.58.0'\n")
    _write_file(driver_node, b"synthetic-node\n", mode=0o755)
    _write_file(driver_cli, b"")
    _write_file(ignored_bytecode, b"path-dependent-cache")
    _write_file(playwright_metadata, b"Name: playwright\nVersion: 1.58.0\n")
    playwright_distribution = _FakeDistribution(
        "playwright",
        "1.58.0",
        {
            "playwright/__init__.py": playwright_init,
            "playwright/driver/node": driver_node,
            "playwright/driver/package/cli.js": driver_cli,
            "playwright/driver/__pycache__/ignored.pyc": ignored_bytecode,
            "playwright-1.58.0.dist-info/METADATA": playwright_metadata,
        },
    )
    return _SyntheticEnvironment(
        root=root,
        purelib=purelib,
        scripts=scripts,
        executable=executable,
        distributions=[demo_distribution, playwright_distribution],
        demo_module=demo_module,
        demo_distribution=demo_distribution,
        playwright_distribution=playwright_distribution,
        driver=driver,
    )


def _add_data_include_distribution(environment: _SyntheticEnvironment) -> None:
    header = (
        environment.root
        / "include"
        / "site"
        / "python3.12"
        / "greenlet"
        / "greenlet.h"
    )
    _write_file(header, b"#define GREENLET_VERSION \"synthetic\"\n")
    environment.distributions.append(
        _FakeDistribution(
            "greenlet",
            "3.2.4",
            {
                "../../../include/site/python3.12/greenlet/greenlet.h": header,
            },
        )
    )


def _rewrite_with_valid_self_hash(path: Path, manifest: dict[str, Any]) -> None:
    without_hash = dict(manifest)
    without_hash.pop("manifest_self_sha256", None)
    manifest["manifest_self_sha256"] = hashlib.sha256(
        _canonical_bytes(without_hash)
    ).hexdigest()
    path.write_bytes(_canonical_bytes(manifest, newline=True))


def test_manifest_is_path_independent_self_hashed_and_includes_empty_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_environment = _synthetic_environment(tmp_path / "first")
    first_environment.activate(monkeypatch)
    first = build_active_environment_manifest()

    second_environment = _synthetic_environment(tmp_path / "different-prefix")
    second_environment.activate(monkeypatch)
    second = build_active_environment_manifest()

    first_direct_url = first_environment.demo_distribution.declared_locations[
        "demo_pkg-1.2.3.dist-info/direct_url.json"
    ].read_bytes()
    second_direct_url = second_environment.demo_distribution.declared_locations[
        "demo_pkg-1.2.3.dist-info/direct_url.json"
    ].read_bytes()
    assert first_direct_url != second_direct_url
    assert first == second
    assert first["kind"] == BROWSERGYM_RUNTIME_MANIFEST_KIND
    assert (
        first["schema_version"]
        == BROWSERGYM_RUNTIME_MANIFEST_SCHEMA_VERSION
    )
    without_hash = dict(first)
    self_hash = without_hash.pop("manifest_self_sha256")
    assert hashlib.sha256(_canonical_bytes(without_hash)).hexdigest() == self_hash
    distributions = {
        distribution["name"]: distribution
        for distribution in first["installed_distributions"]
    }
    assert distributions["demo-pkg"]["content"]["files"] == 5
    assert distributions["demo-pkg"]["excluded_entry_point_launchers"] == [
        {"artifact": "script", "group": "console_scripts", "name": "demo-tool"}
    ]
    assert distributions["playwright"]["content"]["files"] == 4
    assert first["playwright_driver"]["content"]["files"] == 2
    assert first["playwright_driver"]["content"]["bytes"] == len(b"synthetic-node\n")
    assert str(tmp_path) not in _canonical_bytes(first).decode()


def test_data_include_is_a_hashed_path_independent_ancillary_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_environment = _synthetic_environment(tmp_path / "first")
    _add_data_include_distribution(first_environment)
    first_environment.activate(monkeypatch)
    first = build_active_environment_manifest()

    second_environment = _synthetic_environment(tmp_path / "second-longer-prefix")
    _add_data_include_distribution(second_environment)
    second_environment.activate(monkeypatch)
    second = build_active_environment_manifest()

    assert first == second
    assert first["scope"]["ancillary_roots"] == ["data-include"]
    distributions = {
        distribution["name"]: distribution
        for distribution in first["installed_distributions"]
    }
    assert distributions["greenlet"]["content"]["files"] == 1
    assert distributions["greenlet"]["content"]["bytes"] == len(
        b"#define GREENLET_VERSION \"synthetic\"\n"
    )


def test_freeze_is_canonical_verified_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.activate(monkeypatch)
    output = tmp_path / "artifacts" / "runtime.json"

    frozen = freeze_active_environment_manifest(output)
    original = output.read_bytes()
    loaded = load_and_verify_environment_manifest(
        output,
        expected_sha256=frozen["manifest_self_sha256"],
    )

    assert loaded == frozen
    assert original == _canonical_bytes(frozen, newline=True)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_active_environment_manifest(output)
    assert output.read_bytes() == original


def test_loader_rejects_tamper_and_valid_hash_schema_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.activate(monkeypatch)
    output = tmp_path / "runtime.json"
    freeze_active_environment_manifest(output)

    manifest = json.loads(output.read_text())
    manifest["installed_distributions"][0]["version"] = "tampered"
    output.write_bytes(_canonical_bytes(manifest, newline=True))
    with pytest.raises(ValueError, match="self-hash mismatch"):
        load_and_verify_environment_manifest(output, verify_active=False)

    manifest = build_active_environment_manifest()
    manifest["unexpected"] = True
    _rewrite_with_valid_self_hash(output, manifest)
    with pytest.raises(ValueError, match="schema drift"):
        load_and_verify_environment_manifest(output, verify_active=False)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"kind":"first","kind":"second"}\n', "duplicate JSON key"),
        (b'{"kind":NaN}\n', "non-finite JSON number"),
    ],
)
def test_loader_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    output = tmp_path / "runtime.json"
    output.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        load_and_verify_environment_manifest(output, verify_active=False)


@pytest.mark.parametrize("drift", ["missing", "extra"])
def test_active_verification_rejects_missing_or_extra_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.activate(monkeypatch)
    output = tmp_path / "runtime.json"
    freeze_active_environment_manifest(output)
    if drift == "missing":
        environment.distributions.remove(environment.demo_distribution)
    else:
        extra_file = environment.purelib / "extra" / "__init__.py"
        extra_metadata = environment.purelib / "extra-1.dist-info" / "METADATA"
        _write_file(extra_file, b"extra\n")
        _write_file(extra_metadata, b"Name: extra\nVersion: 1\n")
        environment.distributions.append(
            _FakeDistribution(
                "extra",
                "1",
                {
                    "extra/__init__.py": extra_file,
                    "extra-1.dist-info/METADATA": extra_metadata,
                },
            )
        )

    with pytest.raises(ValueError, match="differs from the active environment"):
        load_and_verify_environment_manifest(output)


def test_distribution_discovery_is_scoped_to_active_import_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.activate(monkeypatch)
    external_egg_info = _FakeDistribution(
        "openlocalagent",
        "0.0.1",
        {},
    )
    observed_paths: list[list[str] | None] = []

    def distributions(*, path: list[str] | None = None) -> tuple[_FakeDistribution, ...]:
        observed_paths.append(path)
        if path == [str(environment.purelib.resolve())]:
            return tuple(environment.distributions)
        return (*environment.distributions, external_egg_info)

    monkeypatch.setattr(
        browsergym_runtime_manifest.importlib.metadata,
        "distributions",
        distributions,
    )

    manifest = build_active_environment_manifest()

    assert observed_paths == [[str(environment.purelib.resolve())]]
    assert "openlocalagent" not in {
        distribution["name"]
        for distribution in manifest["installed_distributions"]
    }


@pytest.mark.parametrize(
    ("inventory", "message"),
    [
        ({}, "empty declared file inventory"),
        ({"empty/__pycache__/only.pyc": b"bytecode"}, "no identity-bearing declared files"),
    ],
)
def test_in_root_distribution_requires_identity_bearing_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory: dict[str, bytes],
    message: str,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    declared_locations: dict[str, Path] = {}
    for relative, payload in inventory.items():
        target = environment.purelib / relative
        _write_file(target, payload)
        declared_locations[relative] = target
    environment.distributions.append(
        _FakeDistribution("empty-in-root", "1", declared_locations)
    )
    environment.activate(monkeypatch)

    with pytest.raises(ValueError, match=message):
        build_active_environment_manifest()


def test_editable_direct_url_and_duplicate_normalized_names_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.demo_distribution.direct_url = json.dumps(
        {"dir_info": {"editable": True}, "url": "file:///source"}
    )
    environment.activate(monkeypatch)
    with pytest.raises(ValueError, match="editable distribution 'demo-pkg' is forbidden"):
        build_active_environment_manifest()

    environment.demo_distribution.direct_url = None
    environment.distributions.append(
        _FakeDistribution(
            "demo.pkg",
            "9",
            dict(environment.demo_distribution.declared_locations),
        )
    )
    with pytest.raises(ValueError, match="duplicate normalized installed distribution"):
        build_active_environment_manifest()


def test_package_file_mutation_and_removal_are_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.activate(monkeypatch)
    output = tmp_path / "runtime.json"
    freeze_active_environment_manifest(output)

    environment.demo_module.write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="differs from the active environment"):
        load_and_verify_environment_manifest(output)

    environment.demo_module.unlink()
    with pytest.raises(ValueError, match="missing, escaping, or contains a symlink"):
        build_active_environment_manifest()


def test_declared_symlink_and_escape_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "symlink-environment")
    outside = tmp_path / "outside.py"
    _write_file(outside, b"outside\n")
    environment.demo_module.unlink()
    environment.demo_module.symlink_to(outside)
    environment.activate(monkeypatch)
    with pytest.raises(ValueError, match="symlink"):
        build_active_environment_manifest()

    escape_environment = _synthetic_environment(tmp_path / "escape-environment")
    escaped = tmp_path / "escaped.py"
    _write_file(escaped, b"escaped\n")
    escape_environment.demo_distribution.declared_locations["../escaped.py"] = escaped
    escape_environment.activate(monkeypatch)
    with pytest.raises(ValueError, match="escapes active import roots"):
        build_active_environment_manifest()


def test_undeclared_playwright_driver_file_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    _write_file(environment.driver / "undeclared.js", b"surprise\n")
    environment.activate(monkeypatch)

    with pytest.raises(ValueError, match="does not exactly match"):
        build_active_environment_manifest()


def test_resolved_python_executable_is_path_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    executable_link = environment.root / "python-link"
    executable_link.symlink_to(environment.executable.name)
    environment.activate(monkeypatch)
    monkeypatch.setattr(
        browsergym_runtime_manifest.sys,
        "executable",
        str(executable_link),
    )

    linked = build_active_environment_manifest()
    monkeypatch.setattr(
        browsergym_runtime_manifest.sys,
        "executable",
        str(environment.executable),
    )
    direct = build_active_environment_manifest()

    assert linked == direct
    assert linked["python"]["executable"] == {
        "bytes": len(b"synthetic-python-executable\n"),
        "sha256": hashlib.sha256(b"synthetic-python-executable\n").hexdigest(),
    }


def test_loader_rejects_wrong_expected_self_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.activate(monkeypatch)
    output = tmp_path / "runtime.json"
    freeze_active_environment_manifest(output)

    with pytest.raises(ValueError, match="expected SHA-256 mismatch"):
        load_and_verify_environment_manifest(
            output,
            expected_sha256="0" * 64,
            verify_active=False,
        )


def test_file_modes_are_bound_into_distribution_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.activate(monkeypatch)
    before = build_active_environment_manifest()

    environment.demo_module.chmod(0o600)
    after = build_active_environment_manifest()

    before_demo = before["installed_distributions"][0]["content"]["sha256"]
    after_demo = after["installed_distributions"][0]["content"]["sha256"]
    assert before_demo != after_demo
    assert before["manifest_self_sha256"] != after["manifest_self_sha256"]


def test_manifest_path_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.activate(monkeypatch)
    output = tmp_path / "runtime.json"
    freeze_active_environment_manifest(output)
    link = tmp_path / "runtime-link.json"
    link.symlink_to(output.name)

    with pytest.raises(ValueError, match="symlink"):
        load_and_verify_environment_manifest(link, verify_active=False)


def test_console_launcher_bytes_are_explicitly_outside_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    environment.activate(monkeypatch)
    before = build_active_environment_manifest()
    launcher = environment.scripts / "demo-tool"

    _write_file(launcher, b"#!/a/different/absolute/python\nprint('changed')\n", mode=0o755)
    after = build_active_environment_manifest()

    assert before == after
    assert "not byte-hashed" in before["scope"]["outside_import_roots"]
    assert os.fspath(environment.root) not in _canonical_bytes(before).decode()


def test_active_python_version_console_alias_is_accepted_and_canonicalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    version_suffix = (
        f"{browsergym_runtime_manifest.sys.version_info.major}."
        f"{browsergym_runtime_manifest.sys.version_info.minor}"
    )
    alias_name = f"demo-tool{version_suffix}"
    alias = environment.scripts / alias_name
    _write_file(alias, b"#!/path-dependent/python\n", mode=0o755)
    environment.demo_distribution.declared_locations[
        f"../../../bin/{alias_name}"
    ] = alias
    environment.activate(monkeypatch)

    manifest = build_active_environment_manifest()

    demo = next(
        distribution
        for distribution in manifest["installed_distributions"]
        if distribution["name"] == "demo-pkg"
    )
    assert {
        "artifact": "python-major-minor",
        "group": "console_scripts",
        "name": "demo-tool",
    } in demo["excluded_entry_point_launchers"]


def test_wrong_python_version_console_alias_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _synthetic_environment(tmp_path / "environment")
    wrong_suffix = (
        f"{browsergym_runtime_manifest.sys.version_info.major}."
        f"{browsergym_runtime_manifest.sys.version_info.minor + 1}"
    )
    alias_name = f"demo-tool{wrong_suffix}"
    alias = environment.scripts / alias_name
    _write_file(alias, b"#!/path-dependent/python\n", mode=0o755)
    environment.demo_distribution.declared_locations[
        f"../../../bin/{alias_name}"
    ] = alias
    environment.activate(monkeypatch)

    with pytest.raises(ValueError, match="not an active entry-point launcher"):
        build_active_environment_manifest()
