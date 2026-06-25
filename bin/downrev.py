#! /usr/bin/env python

import re
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path

import toml
import typer
from packaging.specifiers import Specifier
from packaging.version import Version


@dataclass(frozen=True)
@total_ordering
class DownrevDep:
    pkg: str
    version: Version

    def __lt__(self, other: DownrevDep) -> bool:
        return self.version < other.version

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DownrevDep) and self.version == other.version

    def __hash__(self) -> int:
        return hash((self.pkg, self.version))


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
    pyproject_toml: Path,
    uv_lock_path: Path,
) -> list[DownrevDep]:

    pyproject_data = toml.load(pyproject_toml)
    project_deps = pyproject_data.get("project", {}).get("dependencies", [])

    lock_data = parse_uv_lock(uv_lock_path)

    downrev_versions = []

    for dep in project_deps:
        if " " in dep:
            name, specifier_str = dep.split(" ", 1)
            assert re.search(r"^[>=]= ", specifier_str), dep
        else:
            name, specifier_str = (
                dep,
                ">= 0.0.0",
            )

        installed_version_str = lock_data.get(name)
        assert installed_version_str, f"{name} is not yet installed in the .venv/"
        installed_version = Version(installed_version_str)
        specifier = Specifier(specifier_str)

        if installed_version not in specifier:
            downrev_versions.append(DownrevDep(name, installed_version))

    return downrev_versions


def main(
    pyproject_toml: Path = Path("pyproject.toml"),
    uv_lock: Path = Path("uv.lock"),
) -> None:
    downrev_dependencies = find_downrev_dependencies(pyproject_toml, uv_lock)

    if downrev_dependencies:
        print("Dependencies with downrev versions:")
        for dep in downrev_dependencies:
            print(f"{dep}")


if __name__ == "__main__":
    typer.run(main)
