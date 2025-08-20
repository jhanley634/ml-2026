#! /usr/bin/env python

from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from dateutil.parser import parse as date_parse


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


def _parse_date(date_str: str | float) -> str | None:
    """Parses a date string into an ISO 8601-compliant format."""
    assert isinstance(date_str, str | float)
    date_str = str(date_str)
    try:
        clean_date_str = date_str.replace("*", "").strip()
        parsed_date = date_parse(clean_date_str)
        return parsed_date.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


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
    assert tbl
    th = tbl.find_next("tr")
    assert isinstance(th, Tag), type(th)
    column_names = [col.get_text(strip=True) for col in th]
    df.columns = list(map(_downcase, filter(None, column_names)))
    df["user_records"] = df.user_records.apply(_to_int)
    df["breach_date"] = pd.to_datetime(df["breach_date"].apply(_parse_date))
    return df


if __name__ == "__main__":
    url = (
        "https://www.bleepingcomputer.com"
        "/news/security"
        "/hacker-leaks-386-million-user-records-from-18-companies-for-free/"
    )
    print(extract_table(url))
