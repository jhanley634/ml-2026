import unittest
from datetime import datetime
from typing import Any
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

from git.log_analysis import find_daily_counts, get_git_commits, mark_intervals, parse_commit_line

UTC = ZoneInfo("UTC")


class TestGitLogAnalysis(unittest.TestCase):

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa
        self.mock_output = """1617187200 Commit message 1
                            1617190800 Commit message 2
                            1617210000 Commit message 3"""
        self.parsed_commits = [
            (datetime(2021, 4, 1, 9, tzinfo=UTC), "Commit message 1"),
            (datetime(2021, 4, 1, 10, tzinfo=UTC), "Commit message 2"),
            (datetime(2021, 5, 1, 0, tzinfo=UTC), "Commit message 3"),
        ]
        super().__init__(*args, **kwargs)

    def test_get_git_commits(self) -> None:
        with mock.patch("subprocess.run") as mocked_run:
            mocked_run.return_value.stdout = self.mock_output
            commits = get_git_commits()
            self.assertEqual(len(commits), 3)
            for commit in commits:
                self.assertIn(commit, self.mock_output.splitlines())

    def test_parse_commit_line(self) -> None:
        for timestamp, message in self.parsed_commits:
            result = parse_commit_line(f"{int(timestamp.timestamp())} {message}")
            self.assertEqual(result[0], timestamp)
            self.assertEqual(result[1], message)

    def test_mark_intervals(self) -> None:
        activity = mark_intervals(self.parsed_commits)

        expected_activity = {
            datetime(2021, 4, 1, hour=9, tzinfo=UTC): 1,
            datetime(2021, 4, 1, hour=10, tzinfo=UTC): 1,
            datetime(2021, 5, 1, hour=0, tzinfo=UTC): 1,
        }
        self.assertEqual(set(expected_activity.keys()), set(activity.keys()))
        for day in activity:
            self.assertEqual(expected_activity[day], activity[day])

    def test_find_daily_counts(self) -> None:
        sample_activity = {
            datetime(2021, 4, 1, hour=9, tzinfo=UTC): 2,
            datetime(2021, 5, 1, hour=10, tzinfo=UTC): 1,
            datetime(2021, 5, 1, hour=11, tzinfo=UTC): 1,
        }
        expected_data = {
            datetime.combine(datetime(2021, 4, 1, tzinfo=UTC), datetime.min.time(), tzinfo=UTC): 2,
            datetime.combine(datetime(2021, 5, 1, tzinfo=UTC), datetime.min.time(), tzinfo=UTC): 2,
        }
        expected_df = pd.DataFrame(
            {
                "stamp": list(expected_data.keys()),
                "count": list(expected_data.values()),
            },
        ).set_index("stamp")

        result_df = find_daily_counts(sample_activity)

        pd.testing.assert_frame_equal(expected_df, result_df)
