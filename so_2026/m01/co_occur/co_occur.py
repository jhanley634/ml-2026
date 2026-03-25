#! /usr/bin/env python

# Inspired by https://stackoverflow.com/questions/79862438
# /vectorise-coincidences-lookup-between-numpy-arrays


# We have a source, a sample of Americium-241. Nearby, both at a distance R, we have
# a pair of alpha particle detectors, A & B, with high-speed counters (clocks).
# They report decay events both from the source and from background radiation.
# Sometimes they report a coincidence or a co-occurence of a decay.
# We wish to focus just on those, filtering out background events.
#
# The counters give integer timestamps in arbitrary time units.
# Define "co-occurence" as abs difference in stamps ≤ 1.

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray


def generate_decay_events(size: int = 100) -> tuple[NDArray[np.int32], NDArray[np.int32]]:

    rng = np.random.default_rng(seed=42)

    stamps_a = rng.integers(0, 1001, dtype=np.int32, size=size)
    stamps_b = rng.integers(0, 1001, dtype=np.int32, size=size)
    stamps_a.sort()
    stamps_b.sort()
    return stamps_a, stamps_b


def find_coincidences(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    max_delta: int = 1,
) -> Generator[tuple[int, int]]:
    assert len(a) == len(b)
    assert max_delta > 0

    # Find events that co-occur within max_delta.
    # This is a 2-way merge of sorted inputs.
    i, j = 0, 0
    while i < len(a) and j < len(b):
        diff = a[i] - b[j]

        if abs(diff) > max_delta:
            # Outside the coincidence window.
            # We must advance the pointer corresponding to the element that is lagging
            # behind the other, effectively shrinking the window towards the valid range
            # or advancing to match larger timestamps.
            # If a[i] - b[j] > max_delta (a is ahead): b[j] is too old, increment j.
            # If b[j] - a[i] > max_delta (b is ahead): a[i] is too old, increment i.
            if diff > 0:
                j += 1
            else:
                i += 1
        else:
            a_i, b_j = map(int, (a[i], b[j]))
            yield a_i, b_j
            # Advance the pointer for the smaller timestamp to prevent duplicates
            # and ensure we don't get stuck in a loop.
            if a[i] < b[j]:
                i += 1
            else:
                j += 1
