#! /usr/bin/env python

# data from:
# https://data.stackexchange.com/stackoverflow/query/1882532/questions-per-month

# chart posted on:
# https://stackoverflow.com/questions/72695327/sql-display-max-2-employees-per-dept

from pathlib import Path

import matplotlib.dates as mdates
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

IN = Path("~/Downloads/QueryResults.csv").expanduser()
OUT = Path("~/Desktop/monthly-questions.png").expanduser()


def plot(
    infile: Path = IN,
    outfile: Path = OUT,
) -> None:
    df = (
        pl.read_csv(infile)
        .with_columns(
            [
                pl.col("Created").str.slice(0, 10).cast(pl.Date).alias("month"),
                pl.col("").alias("questions"),
            ]
        )
        .select(["month", "questions"])
    )
    df = df.with_columns(pl.col("questions").rolling_mean(window_size=12).alias("EMA_questions"))
    print(df)

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(data=df.to_pandas(), x="month", y="questions", marker="o")
    sns.lineplot(
        data=df.to_pandas(),
        x="month",
        y="EMA_questions",
        linestyle="--",
        color="purple",
        ax=ax,
        label="moving avg.",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Number of Questions")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(outfile)


if __name__ == "__main__":
    plot()
