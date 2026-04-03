#! /usr/bin/env python


from pathlib import Path

import toml
import typer
from packaging.specifiers import Specifier
from packaging.version import Version


def parse_uv_lock(in_file: Path) -> dict[str, str]:
    uv_lock_data = toml.load(in_file)
    packages = uv_lock_data["package"]
    uv_lock_versions = {}
    assert isinstance(packages, list), type(packages)
    for pkg in packages:
        name = pkg["name"]
        version = pkg["version"]
        uv_lock_versions[name] = version
    return uv_lock_versions


def find_downrev_dependencies(
    pyproject_toml_path: Path,
    uv_lock_path: Path,
) -> list[Version]:

    pyproject_data = toml.load(pyproject_toml_path)
    dependencies = pyproject_data.get("project", {}).get("dependencies", [])

    lock_data = parse_uv_lock(uv_lock_path)

    downrev_versions = []

    for dep in dependencies:
        if ">=" in dep and dep.startswith("package"):
            parts = dep.split(">=")
            if len(parts) == 2:
                package_name = parts[0]
                version_spec = parts[1]
                specifier = Specifier(f">= {version_spec}")
                # Find the locked version that satisfies the specifier
                locked_version = lock_data.get(package_name)
                if locked_version and Version(locked_version) in specifier:
                    downrev_versions.append(Version(locked_version))

    return downrev_versions


def main(
    pyproject: Path = Path("pyproject.toml"),
    uv_lock: Path = Path("uv.lock"),
) -> None:
    downrev_dependencies = find_downrev_dependencies(pyproject, uv_lock)

    if downrev_dependencies:
        print("Dependencies with downrev versions:")
        for dep in downrev_dependencies:
            print(f"{dep}")


if __name__ == "__main__":
    typer.run(main)
