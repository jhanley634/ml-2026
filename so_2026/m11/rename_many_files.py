#! /usr/bin/env python

# cf https://stackoverflow.com/staging-ground/80015251


import re
from pathlib import Path
from typing import TYPE_CHECKING

import cappa

if TYPE_CHECKING:
    from collections.abc import Generator

TEMP = Path("/tmp")
FILE_LIST = TEMP / "file_list.txt"


def get_file_paths(folder: Path = Path()) -> Generator[Path]:
    """Same as:  find . -type f"""
    for path in folder.rglob("*"):
        if path.is_file():
            yield path


bracket_re = re.compile(r"\[[\w=]+\]|deps/tracker-fcf")


def main(folder: Path = Path()) -> None:

    # Has same effect as:  find . -type f | sort  > file_list.txt
    if not FILE_LIST.exists():
        with FILE_LIST.open("w") as fout:
            for path in sorted(get_file_paths(folder)):
                fout.write(f"{path}\n")

    for path in FILE_LIST.read_text()[:-1].split("\n"):
        if bracket_re.search(path):
            print(path)


if __name__ == "__main__":
    main()
