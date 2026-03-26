import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np
    from numpy.typing import NDArray

assert sys.version_info >= (3, 13)


@dataclass
class Event:
    stamp: int  # timestamp in arbitrary units
    det_id: int  # detector ID; zero-origin
    _order: tuple[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_order", (self.stamp, self.det_id))

    def __hash__(self) -> int:
        return hash(self._order)

    def __lt__(self, other: Event) -> bool:
        return self._order < other._order

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Event):
            return self._order == other._order
        return False


def merge_event_streams(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
) -> Generator[Event]:
    """
    2-way merge.
    We generate monotonically increasing Events from detectors a & b.
    """
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            yield Event(a[i], det_id=0)
            i += 1
        else:
            yield Event(b[j], det_id=1)
            j += 1

    # Having consumed one of the inputs, we will now emit all from the other one.

    while i < len(a):
        yield Event(a[i], det_id=0)
        i += 1
    while j < len(b):
        yield Event(b[j], det_id=1)
        j += 1
