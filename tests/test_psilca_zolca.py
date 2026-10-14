"""Native-import regressions using invented data only, never licensed inputs."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from sense.data import psilca_zolca, zolca_reader
from sense.data.psilca import (
    combine_psilca_css_and_extract_social_exchanges,
    validate_psilca_source,
)

ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / "src/sense/resources/zolca/reader.jar"
FIXTURE = ROOT / "tests/fixtures/zolca/synthetic_fixture.jar"


def _fixture(root, mode=None):
    try:
        java = zolca_reader.java_executable()
    except RuntimeError as error:
        pytest.skip(str(error))
    root.mkdir(parents=True)
    # Relative classpath + Base64 arguments preserve Unicode on Windows.
    shutil.copyfile(READER, root / "reader.jar")
    shutil.copyfile(FIXTURE, root / "fixture.jar")
    command = [
        java,
        "-cp",
        os.pathsep.join(["reader.jar", "fixture.jar"]),
        "SyntheticSocialDatabase",
        base64.b64encode(str(root).encode()).decode(),
    ]
    if mode:
        command.append(mode)
    result = subprocess.run(command, cwd=root, capture_output=True, timeout=30)
    assert result.returncode == 0, "Invented fixture creation failed"
    return root / "synthetic.zolca"


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    return _fixture(tmp_path_factory.mktemp("zolca") / "café 工場 with spaces")


def _xml_fixture(folder):
    folder.mkdir()
    assessments = [
        [
            ("Synthetic safety; high risk", 2.0),
            ("Synthetic training; high opportunity", 0.125),
            ("Synthetic training; high opportunity", 0.125),
        ],
        [
            ("Synthetic safety; no data", 0.0),
            ("Synthetic training; low opportunity", -1.25),
            ("Synthetic safety; no data", None),
        ],
    ]
    for i, values in enumerate(assessments, 101):
        exchanges = []
        for name, hours in values:
            amount = f' meanValue="{hours}"' if hours is not None else ""
            exchanges.append(
                f'<exchange name="{name}" unit="h"{amount} category="Invented workers" subCategory="Synthetic safety and training"><outputGroup>4</outputGroup></exchange>'
            )
        name = "Invented product A" if i == 101 else "Invented product B"
        country = "AA" if i == 101 else "BB"
        xml = f'<ecoSpold><dataset><metaInformation><processInformation><referenceFunction name="{name}"/><geography location="{country}"/></processInformation></metaInformation><flowData>{"".join(exchanges)}</flowData></dataset></ecoSpold>'
        (folder / f"process_00000000-0000-4000-8000-000000000{i}.xml").write_text(xml)


def test_native_matches_xml_preserves_original_values_and_product_labels(
    archive, tmp_path, monkeypatch
):
    before = archive.read_bytes()
    resources = tmp_path / "reader café 工場"
    resources.mkdir()
    (resources / "zolca").mkdir()
    shutil.copyfile(READER, resources / "zolca/reader.jar")
    monkeypatch.setattr(zolca_reader, "files", lambda _: resources)
    native = tmp_path / "native.parquet"
    psilca_zolca.extract_zolca(archive, native, tmp_path / "native.json", tmp_path / "zeros.json")
    xml = tmp_path / "xml"
    _xml_fixture(xml)
    combine_psilca_css_and_extract_social_exchanges(
        str(xml), str(tmp_path / "xml.parquet"), str(tmp_path / "xml.json")
    )
    left = pq.read_table(native).to_pandas()
    right = pq.read_table(tmp_path / "xml.parquet").to_pandas()
    pd.testing.assert_frame_equal(left, right)
    assert json.loads((tmp_path / "native.json").read_text()) == json.loads(
        (tmp_path / "xml.json").read_text()
    )
    assert len(left) == 6 and left.mean_value.isna().sum() == 1
    assert left.assessment_type.tolist().count("opportunity") == 3
    assert left.reference_name.tolist()[:3] == ["Invented product A"] * 3
    assert [row["Reason"] for row in json.loads((tmp_path / "zeros.json").read_text())] == [
        "zero",
        "negative",
    ]
    assert archive.read_bytes() == before


@pytest.mark.parametrize("mode", ["invalid-risk", "library"])
def test_unsupported_assessments_preserve_previous_artifacts(tmp_path, mode):
    source = _fixture(tmp_path / "fixture", mode)
    target = tmp_path / "old.parquet"
    sectors = tmp_path / "sectors.json"
    target.write_bytes(b"previous cache")
    sectors.write_bytes(b"previous sectors")
    with pytest.raises(ValueError, match="Unsupported or inconsistent"):
        psilca_zolca.extract_zolca(source, target, sectors)
    assert target.read_bytes() == b"previous cache"
    assert sectors.read_bytes() == b"previous sectors"
    assert not list(tmp_path.glob("*.tmp*"))


def test_writer_failure_closes_reader_and_preserves_cache(archive, tmp_path, monkeypatch):
    target = tmp_path / "old.parquet"
    target.write_bytes(b"previous cache")
    closed = []
    original = psilca_zolca.reader_lines

    def wrapped(path):
        try:
            yield from original(path)
        finally:
            closed.append(True)

    def fail(*args):
        raise OSError("synthetic full disk")

    monkeypatch.setattr(psilca_zolca, "reader_lines", wrapped)
    monkeypatch.setattr(psilca_zolca, "_write_parquet_batch", fail)
    with pytest.raises(OSError, match="synthetic full disk"):
        psilca_zolca.extract_zolca(archive, target, tmp_path / "sectors.json")
    assert target.read_bytes() == b"previous cache"
    assert closed == [True]


@pytest.mark.parametrize("name", ["../seg0/c10.dat", "/seg0/c10.dat", "seg0\\c10.dat"])
def test_unsafe_native_backups_are_rejected_before_java(tmp_path, name):
    source = tmp_path / "bad.zolca"
    with zipfile.ZipFile(source, "w") as out:
        out.writestr("service.properties", b"invented")
        out.writestr("seg0/c10.dat", b"invented")
        out.writestr(name, b"invented")
    with pytest.raises(ValueError, match="Unsafe"):
        zolca_reader.archive_fingerprint(source)


def test_library_zip_and_invalid_backup_are_rejected(tmp_path):
    source = tmp_path / "library.zip"
    source.write_bytes(b"invented")
    with pytest.raises(ValueError, match="not the library ZIP"):
        validate_psilca_source(source)
    source = tmp_path / "invalid.zolca"
    source.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a valid ZIP"):
        validate_psilca_source(source)


def test_archive_fingerprint_tracks_entries_not_only_size_and_mtime(archive, tmp_path):
    copy = tmp_path / "copy.zolca"
    shutil.copyfile(archive, copy)
    initial = zolca_reader.archive_fingerprint(copy)
    with zipfile.ZipFile(copy, "a") as out:
        out.writestr("invented-metadata.txt", b"changed")
    assert zolca_reader.archive_fingerprint(copy) != initial


def test_missing_or_old_java_has_actionable_error(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(zolca_reader.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="openjdk=21"):
        zolca_reader.java_executable()
    monkeypatch.setattr(zolca_reader.shutil, "which", lambda _: "java")
    monkeypatch.setattr(
        zolca_reader.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", 'openjdk version "17.0.1"'),
    )
    with pytest.raises(RuntimeError, match="Java 21"):
        zolca_reader.java_executable()


def test_native_cache_reused_without_java_and_invalidated_on_change(archive, tmp_path, monkeypatch):
    from sense.config import SenseConfig
    from sense.preparation import _prepare_psilca_data

    cache = tmp_path / "assessments.parquet"
    args = (archive, SenseConfig(), cache, tmp_path / "sectors.json", tmp_path / "zeros.json")
    _prepare_psilca_data(*args)
    monkeypatch.setattr(
        zolca_reader, "java_executable", lambda: pytest.fail("Cached import started Java")
    )
    _prepare_psilca_data(*args)
    cache.write_bytes(b"corrupt cached artifact")
    with pytest.raises(pytest.fail.Exception, match="Cached import started Java"):
        _prepare_psilca_data(*args)
    assert cache.read_bytes() == b"corrupt cached artifact"


def test_legacy_study_configuration_still_resolves_xml_directory():
    from case_studies.hydrogen.workflow.case_config import psilca_source

    assert psilca_source({"paths": {"psilca_dir": "old/xml"}}).name == "xml"
    assert psilca_source({"paths": {"psilca_path": "new.zolca"}}).name == "new.zolca"
    with pytest.raises(ValueError, match="do not set both"):
        psilca_source({"paths": {"psilca_path": "new.zolca", "psilca_dir": "old/xml"}})


def test_bundled_public_reader_matches_sources_and_retains_apache_notices():
    import hashlib

    with zipfile.ZipFile(READER) as artifact:
        recorded = artifact.read("META-INF/sense-reader-sources.sha256").decode()
        expected = "".join(
            hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n"
            for path in sorted(READER.parent.glob("*.java"))
        )
        assert recorded == expected
        names = artifact.namelist()
        assert any(name.endswith("LICENSE") or name.endswith("LICENSE.txt") for name in names)
        assert any(name.endswith("NOTICE") or name.endswith("NOTICE.txt") for name in names)
        assert "SenseZolcaReader.class" in names
        assert "org/apache/derby/iapi/jdbc/AutoloadedDriver.class" in names
        assert not any("olca" in name.lower() and "SenseZolca" not in name for name in names)


def test_truncated_pipe_cannot_publish_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        psilca_zolca, "reader_lines", lambda _: (fields for fields in [["BEGIN", "1"]])
    )
    target = tmp_path / "old.parquet"
    target.write_bytes(b"previous cache")
    with pytest.raises(ValueError, match="No partial cache"):
        psilca_zolca.extract_zolca(tmp_path / "unused.zolca", target, tmp_path / "sectors.json")
    assert target.read_bytes() == b"previous cache"


def test_old_java_home_does_not_hide_java_21_on_path(tmp_path, monkeypatch):
    home = tmp_path / "old-java"
    binary = home / "bin" / ("java.exe" if os.name == "nt" else "java")
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setenv("JAVA_HOME", str(home))
    monkeypatch.setattr(zolca_reader.shutil, "which", lambda _: "new-java")

    def version(command, **kwargs):
        number = "21.0.1" if command[0] == "new-java" else "17.0.1"
        return subprocess.CompletedProcess(command, 0, "", f'openjdk version "{number}"')

    monkeypatch.setattr(zolca_reader.subprocess, "run", version)
    assert zolca_reader.java_executable() == "new-java"
