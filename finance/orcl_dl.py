#! /usr/bin/env python

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

if TYPE_CHECKING:
    from collections.abc import Iterable


def round_to_cents(prices: Iterable[float]) -> list[float]:
    return [round(price, 2) for price in prices]


def main(ticker: str = "ORCL") -> None:
    start = "2025-03-01"
    end = "2026-06-04"  # one day beyond, similar to range()

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df = df.rename(columns={"Adj Close": "adj_close"})
    df = df[["Open", "High", "Low", "Close", "adj_close", "Volume"]]

    assert isinstance(df.columns, pd.MultiIndex)
    df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df = df.rename(columns={"index": "date"})

    price_cols = ["Open", "High", "Low", "Close", "adj_close"]
    df[price_cols] = df[price_cols].apply(
        lambda x: x.apply(lambda y: round(y, 2)),
        axis=0,
    )
    print(df)

    folder = Path("~/Desktop").expanduser()
    out_path = folder / f"{ticker}.csv"
    df.to_csv(out_path, date_format="%Y-%m-%d", index=False)


if __name__ == "__main__":
    main()
