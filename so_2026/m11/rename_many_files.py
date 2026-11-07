#! /usr/bin/env python

# cf https://stackoverflow.com/staging-ground/80015251


import re
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from collections.abc import Generator

TEMP = Path("/tmp")
FILE_LIST = TEMP / "file_list.txt"


def get_file_paths(folder: Path) -> Generator[Path]:
    """Same as:  find . -type f"""
    for path in folder.rglob("*"):
        if path.is_file():
            yield path


def store_file_paths(folder: Path, dst_file: Path) -> None:
    """Has same effect as:  find . -type f | sort  > file_list.txt"""
    with dst_file.open("w", encoding="utf8") as fout:
        for path in sorted(get_file_paths(folder)):
            fout.write(f"{path}\n")


bracket_re = re.compile(r"\[[\w=]+\]")


def main(folder: Path = Path()) -> None:
    if not FILE_LIST.exists():
        store_file_paths(folder, FILE_LIST)

    for path in FILE_LIST.read_text()[:-1].split("\n"):
        if bracket_re.search(path):
            new = bracket_re.sub("", path)
            print(path, "\t-->   ", new)
            Path(path).rename(new)


if __name__ == "__main__":
    typer.run(main)
