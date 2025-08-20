#! /usr/bin/env python

from io import StringIO
from pprint import pp

import pandas as pd
import requests
from bs4 import BeautifulSoup


def _downcase(s: str) -> str:
    return s.lower().replace(" ", "_").replace("?", "")


def extract_table_from_bleepingcomputer(url: str) -> pd.DataFrame:
    headers = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)")}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    df = pd.read_html(StringIO(response.text))[0]

    soup = BeautifulSoup(StringIO(response.text), "lxml")
    tbl = soup.find_all("table")[0]
    th = tbl.find_next("tr")
    column_names = list(filter(None, (col.get_text(strip=True) for col in th)))
    df.columns = map(_downcase, column_names)
    return df[:-1]


if __name__ == "__main__":
    url = (
        "https://www.bleepingcomputer.com/news/security"
        "/hacker-leaks-386-million-user-records-from-18-companies-for-free/"
    )
    print(extract_table_from_bleepingcomputer(url))
