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
    from numpy.typing import NDArray


def generate_decay_events(size: int = 100) -> tuple[NDArray[np.int32], NDArray[np.int32]]:

    rng = np.random.default_rng(seed=42)

    stamps_a = rng.integers(0, 1001, dtype=np.int32, size=size)
    stamps_b = rng.integers(0, 1001, dtype=np.int32, size=size)
    return stamps_a, stamps_b
