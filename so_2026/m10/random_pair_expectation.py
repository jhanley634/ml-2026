# see https://codereview.stackexchange.com/a/303554
# https://codereview.stackexchange.com/questions/303541/leetcode-twosum
# YT math: https://www.youtube.com/shorts/Pny70rNPJLk

import unittest
from random import randrange
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from collections.abc import Generator


def random_pair(n: int) -> tuple[int, int]:
    a, b = sorted(
        (
            randrange(n),
            randrange(n),
        ),
    )
    return a, b


N = 1_000_000

TRIALS = 100_000


def _gen(n: int, trials: int = TRIALS) -> Generator[dict[str, int]]:
    for i in range(trials):
        a, b = random_pair(n)
        yield {"i": i, "a": a, "b": b}


def plot(n: int = N) -> None:
    df = pd.DataFrame(_gen(n))

    plt.figure()
    sns.kdeplot(np.array(df["a"]), fill=True, color="blue", label="KDE of a")
    sns.kdeplot(np.array(df["b"]), fill=True, color="red", label="KDE of b")
    plt.title("kernel density estimate of random pairs")
    plt.xlabel("value")
    plt.ylabel("density")
    plt.legend()
    plt.show()

    print(df.describe())


class RandomPairTest(unittest.TestCase):
    def test_random_pair(self) -> None:
        for _ in range(TRIALS):
            a, b = random_pair(N)
            self.assertLessEqual(a, b)
            assert a in range(N)
            assert b in range(N)


if __name__ == "__main__":
    plot()
