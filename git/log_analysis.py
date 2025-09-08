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
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


def get_git_commits() -> list[str]:
    """Retrieve git commit logs with timestamps."""
    result = subprocess.run(
        ["git", "log", "--pretty=format:%at %s"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def parse_commit_line(line: str) -> tuple[datetime, str]:
    """Parse a line from git log to extract timestamp and message."""
    timestamp_str, commit_message = line.split(" ", 1)
    timestamp = datetime.fromtimestamp(int(timestamp_str), tz=UTC)
    return timestamp, commit_message


def mark_intervals(commit_timestamps: list[tuple[datetime, str]]) -> dict[date, list[int]]:
    """Mark active intervals based on commit timestamps."""
    interval_count = 48
    interval_duration = timedelta(minutes=30)

    activity = {}

    for timestamp_tuple in commit_timestamps:
        timestamp = timestamp_tuple[0]  # Access the datetime from the tuple
        day_key = timestamp.date()

        if day_key not in activity:
            activity[day_key] = [0] * interval_count

        start_of_day = datetime.combine(day_key, datetime.min.time(), tzinfo=UTC)
        elapsed_time = timestamp - start_of_day
        interval_index = int(elapsed_time // interval_duration)

        activity[day_key][interval_index] = 1

    return activity


def calculate_daily_hours(activity: dict[date, list[int]]) -> dict[date, float]:
    """Calculate total hours of activity per day."""
    daily_hours = {}

    for day, intervals in activity.items():
        active_intervals = sum(intervals)
        # Convert active intervals to hours (48 intervals per day)
        daily_hours[day] = active_intervals * (60 / 2) / 60

    return daily_hours


def main() -> None:
    commits = get_git_commits()
    commit_pairs = [parse_commit_line(line) for line in commits]

    activity = mark_intervals(commit_pairs)
    daily_hours = calculate_daily_hours(activity)

    for day, hours in sorted(daily_hours.items()):
        print(f"{day}: {hours:.2f} hours")


if __name__ == "__main__":
    main()
