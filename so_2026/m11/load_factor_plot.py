#! /usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from so_2026.m11.load_factor import HashSet


def evaluate_load_factors():
    results = []
    load_factors = [i * 0.01 for i in range(60, 96)]
    initial_num_elts = 1000  # Adjust as needed

    for load_factor in load_factors:
        hash_set = HashSet(initial_num_elts, load_factor)
        for _ in range(int(initial_num_elts * load_factor)):
            hash_set.put(f"key{_}", verbose=False)

        # Collect data for chains of length 3 and 4
        chain_3_count = hash_set.chain_length.get(3, 0)
        chain_4_count = hash_set.chain_length.get(4, 0)
        results.append((load_factor, chain_3_count, chain_4_count))

    df = pd.DataFrame(results, columns=["Load Factor", "Chain Length 3", "Chain Length 4"])
    return df


def plot_chains(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Load Factor", y="Chain Length 3", marker="o", label="Chain Length 3")
    sns.lineplot(data=df, x="Load Factor", y="Chain Length 4", marker="o", label="Chain Length 4")
    plt.title("Chains of Length 3 and 4 as Load Factor Approaches 90%")
    plt.xlabel("Load Factor")
    plt.ylabel("Chain Count")
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


def main() -> None:
    plot_chains(evaluate_load_factors())


if __name__ == "__main__":
    main()
