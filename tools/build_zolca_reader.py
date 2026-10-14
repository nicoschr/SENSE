"""Rebuild the public-only PSILCA reader (Java 21 and Apache Derby 10.17.1.0).

Uses pinned, verified Maven Central artifacts. No PSILCA data is needed or read.
Normal installation uses the bundled JAR and does not run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "src/sense/resources/zolca"
DERBY_VERSION = "10.17.1.0"
ARTIFACTS = {
    "derby": "764e4c133f860a8876e835ef2306efecad27a742d27f05bc4bce669432c6b397",
    "derbyshared": "fb68bab30785375c1f8cb5e9583b349750f8904d6f7ee15ce8e1c30b6be30ef4",
}


def version_output(result: subprocess.CompletedProcess[str]) -> str:
    """Retain version information without echoing inherited JVM environment options."""
    return "\n".join(
        line
        for line in (result.stdout + result.stderr).splitlines()
        if not line.startswith(("Picked up ", "NOTE: Picked up "))
    ).strip()


def build(java: str, javac: str | None, dependencies: Path | None, compiler_jar: Path | None):
    with tempfile.TemporaryDirectory(prefix="sense-reader-build-") as folder:
        temporary = Path(folder)
        jars = []
        for name, digest in ARTIFACTS.items():
            jar = temporary / f"{name}.jar"
            local = dependencies / f"{name}-{DERBY_VERSION}.jar" if dependencies else None
            if local and local.is_file():
                shutil.copyfile(local, jar)
            else:
                url = f"https://repo.maven.apache.org/maven2/org/apache/derby/{name}/{DERBY_VERSION}/{name}-{DERBY_VERSION}.jar"
                with urllib.request.urlopen(url, timeout=120) as response, jar.open("wb") as out:
                    shutil.copyfileobj(response, out)
            if hashlib.sha256(jar.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch for public dependency {name}")
            jars.append(jar)
        classes = temporary / "classes"
        classes.mkdir()
        if compiler_jar:
            command = [java, "-jar", str(compiler_jar), "-21"]
            compiler_version_command = [java, "-jar", str(compiler_jar), "-version"]
        else:
            command = [javac or "javac", "--release", "21"]
            compiler_version_command = [javac or "javac", "-version"]
        compiler_version = subprocess.run(
            compiler_version_command, capture_output=True, text=True, check=True, timeout=15
        )
        java_version = subprocess.run(
            [java, "-version"], capture_output=True, text=True, check=True, timeout=15
        )
        subprocess.run(
            [
                *command,
                "-encoding",
                "UTF-8",
                "-d",
                str(classes),
                *map(str, sorted(RESOURCES.glob("*.java"))),
            ],
            check=True,
        )
        contents = {}
        for jar in jars:
            with zipfile.ZipFile(jar) as archive:
                for entry in archive.infolist():
                    name = entry.filename
                    if (
                        entry.is_dir()
                        or name == "module-info.class"
                        or name.upper().endswith((".SF", ".RSA", ".DSA"))
                        or name == "META-INF/MANIFEST.MF"
                    ):
                        continue
                    data = archive.read(entry)
                    if name in contents and contents[name] != data:
                        # Preserve separate Apache notices when module paths overlap.
                        if name.endswith(("LICENSE", "NOTICE", "LICENSE.txt", "NOTICE.txt")):
                            name = "META-INF/" + jar.stem + "-" + Path(name).name
                        else:
                            raise ValueError(f"Conflicting public dependency resource: {name}")
                    contents[name] = data
        for path in classes.rglob("*.class"):
            contents[path.relative_to(classes).as_posix()] = path.read_bytes()
        contents["META-INF/sense-reader-sources.sha256"] = "".join(
            hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n"
            for path in sorted(RESOURCES.glob("*.java"))
        ).encode("ascii")
        contents["META-INF/MANIFEST.MF"] = (
            b"Manifest-Version: 1.0\r\nMain-Class: SenseZolcaReader\r\n\r\n"
        )
        provenance = {
            "schema": 1,
            "target_java_release": 21,
            "compiler": version_output(compiler_version),
            "java_runtime": version_output(java_version),
            "compiler_kind": "ecj" if compiler_jar else "javac",
            "compiler_sha256": (
                hashlib.sha256(compiler_jar.read_bytes()).hexdigest() if compiler_jar else None
            ),
            "compile_options": ["-21" if compiler_jar else "--release=21", "-encoding=UTF-8"],
            "sources": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(RESOURCES.glob("*.java"))
            },
            "dependencies": {
                name: {"version": DERBY_VERSION, "sha256": digest}
                for name, digest in ARTIFACTS.items()
            },
        }
        contents["META-INF/sense-reader-build.json"] = json.dumps(
            provenance, sort_keys=True, indent=2
        ).encode("utf-8")
        artifact = RESOURCES / "reader.jar"
        staged = temporary / "reader.jar"
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        # Stage in the destination filesystem before replacement, even when the
        # system's temporary directory is on another volume.
        with tempfile.NamedTemporaryFile(dir=RESOURCES, suffix=".tmp", delete=False) as handle:
            destination_temp = Path(handle.name)
        try:
            shutil.copyfile(staged, destination_temp)
            os.replace(destination_temp, artifact)
        finally:
            destination_temp.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=RESOURCES, suffix=".tmp", delete=False
        ) as handle:
            metadata_temp = Path(handle.name)
            json.dump(
                {
                    **provenance,
                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                },
                handle,
                indent=2,
            )
        try:
            os.replace(metadata_temp, RESOURCES / "reader-build.json")
        finally:
            metadata_temp.unlink(missing_ok=True)
        print(
            f"Public reader built: {artifact.name}, {artifact.stat().st_size} bytes, SHA256 {hashlib.sha256(artifact.read_bytes()).hexdigest()}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java", default="java")
    parser.add_argument("--javac")
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument(
        "--compiler-jar",
        type=Path,
        help="Optional development compiler for a JRE-only installation",
    )
    args = parser.parse_args()
    build(args.java, args.javac, args.dependency_dir, args.compiler_jar)
