#! /usr/bin/env python

from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup


def _to_int(value: str) -> int:
    """Converts a string like '5.8 Million' into the integer 5_800_000."""
    value = value.strip().lower()
    if "million" in value:
        number = float(value.split()[0]) * 1_000_000
    elif "thousand" in value:
        number = float(value.split()[0]) * 1_000
    else:
        number = float(value)
    return int(number)


def _downcase(s: str) -> str:
    s = s.lower().replace(" ", "_")
    s = s.replace("reported_", "")
    return s.replace("?", "")


def extract_table(url: str) -> pd.DataFrame:
    headers = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)")}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    df = pd.read_html(StringIO(response.text))[0][:-1]
    df = df.drop(0)

    soup = BeautifulSoup(StringIO(response.text), "lxml")
    tbl = soup.find_all("table")[0]
    th = tbl.find_next("tr")
    column_names = list(filter(None, (col.get_text(strip=True) for col in th)))
    df.columns = map(_downcase, column_names)
    df["user_records"] = df.user_records.apply(_to_int)
    return df


if __name__ == "__main__":
    url = (
        "https://www.bleepingcomputer.com"
        "/news/security"
        "/hacker-leaks-386-million-user-records-from-18-companies-for-free/"
    )
    print(extract_table(url))
