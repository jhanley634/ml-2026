#! /usr/bin/env python

"""
Displays "hours spent coding" on a daily basis, from git logs.

Divide the day into 48 equal length intervals.
Each time a commit happens, mark the corresponding interval "active".
Report daily total hours of activity on the checked out feature branch.

NB: A simple `git log` will not suffice, as we frequently
use `git --amend --no-edit`. Each timestamped amendment
will contribute to the activity record.
"""

import subprocess
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

UTC = ZoneInfo("UTC")


def get_git_commits() -> list[str]:
    """Retrieve git commit logs with timestamps."""
    result = subprocess.run(
        ["git", "reflog", "--pretty=format:%at %gs"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def parse_commit_line(line: str) -> tuple[datetime, str]:
    """Parse a line from git log to extract timestamp and message."""
    timestamp_str, commit_message = line.split(" ", 1)
    timestamp = datetime.fromtimestamp(int(timestamp_str), tz=UTC)
    return timestamp, commit_message


def mark_intervals(commit_timestamps: list[tuple[datetime, str]]) -> dict[datetime, int]:
    """Mark each active interval with its number of commits."""
    activity: dict[datetime, int] = defaultdict(int)

    for stamp, _ in commit_timestamps:
        interval_start = datetime(
            year=stamp.year,
            month=stamp.month,
            day=stamp.day,
            hour=stamp.hour,
            minute=(stamp.minute // 30) * 30,  # half-hour intervals
            second=0,
            tzinfo=UTC,
        )
        activity[interval_start] += 1

    return activity


def find_daily_counts(activity: dict[datetime, int]) -> pd.DataFrame:
    # Find per-interval counts.
    df = pd.DataFrame(
        {
            "stamp": list(activity.keys()),
            "count": list(activity.values()),
        },
    )
    df = df.set_index("stamp")

    # Now resample at daily frequency, summing up each day's counts.
    ret = df.resample("D").sum()
    ret = ret[ret["count"] > 0]
    ret.index.freq = None
    return ret


def main() -> None:
    commit_pairs = [parse_commit_line(line) for line in get_git_commits()]
    print(find_daily_counts(mark_intervals(commit_pairs)))


if __name__ == "__main__":
    main()
