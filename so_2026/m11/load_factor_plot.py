#! /usr/bin/env python


from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from so_2026.m11.load_factor import HashSet

if TYPE_CHECKING:
    from collections.abc import Generator


def evaluate_load_factors() -> Generator[dict[str, float]]:
    load_factors = [i * 0.01 for i in range(60, 99)]
    initial_num_elts = 1000

    for load_factor in load_factors:
        hash_set = HashSet(initial_num_elts, load_factor)
        for i in range(initial_num_elts):
            hash_set.put(f"key{i}")

        for length in range(2, 10):
            count = hash_set.chain_length.get(length, 0)
            yield {
                "load_factor": load_factor,
                "chain_length": length,
                "chain_count": count,
            }


def plot_chains() -> None:
    df = pd.DataFrame(evaluate_load_factors())
    print(df)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="load_factor", y="chain_count", hue="chain_length", marker="o")
    plt.title("Chain Counts as Load Factor Approaches 98%")
    plt.xlabel("Load Factor")
    plt.ylabel("Chain Count")
    plt.ylim(bottom=0)
    plt.legend(title="Chain Length")
    plt.show()


def main() -> None:
    plot_chains()


if __name__ == "__main__":
    main()
