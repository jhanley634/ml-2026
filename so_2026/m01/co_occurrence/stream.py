import sys
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np
    from numpy.typing import NDArray

assert sys.version_info >= (3, 13)


@dataclass(order=True, frozen=True)
class Event:
    stamp: int  # timestamp in arbitrary units
    det_id: int  # detector ID; zero-origin


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


def window(
    events: Generator[Event],
    window_size: int = 3,
) -> Generator[tuple[Event, dict[int, Event]]]:

    stamp_to_event: dict[int, Event] = {}
    recent: deque[Event] = deque()

    for event in events:
        yield event, stamp_to_event
        stamp_to_event[event.stamp] = event
        recent.append(event)

        if len(recent) > window_size:
            ancient = recent.popleft()
            if ancient.stamp in stamp_to_event:
                del stamp_to_event[ancient.stamp]

        assert len(stamp_to_event) <= window_size
        assert len(recent) <= window_size
