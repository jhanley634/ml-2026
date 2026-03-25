# from https://codereview.stackexchange.com/a/301010/find-median-of-two-sorted-arrays


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


def merge(lst1: Sequence[float], lst2: Sequence[float]) -> Generator[float]:
    """
    Creates a generator yielding sorted,
    merged contents of two sorted input sequences / lists.
    """

    i, j = 0, 0

    while i < len(lst1) or j < len(lst2):
        if i >= len(lst1) or (j < len(lst2) and lst1[i] >= lst2[j]):
            yield lst2[j]
            j += 1
        else:
            yield lst1[i]
            i += 1
