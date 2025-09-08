import unittest
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

from git.log_analysis import (
    calculate_daily_hours,
    get_git_commits,
    mark_intervals,
    parse_commit_line,
)

UTC = ZoneInfo("UTC")
TZ = ZoneInfo("America/Los_Angeles")


class TestGitLogAnalysis(unittest.TestCase):

    def __init__(self) -> None:
        self.mock_output = """1617187200 Commit message 1
                            1617190800 Commit message 2
                            1617210000 Commit message 3"""
        self.parsed_commits = [
            (datetime(2021, 4, 1, 9, tzinfo=UTC), "Commit message 1"),
            (datetime(2021, 4, 1, 10, tzinfo=UTC), "Commit message 2"),
            (datetime(2021, 5, 1, 0, tzinfo=UTC), "Commit message 3"),
        ]

    def test_get_git_commits(self) -> None:
        with mock.patch("subprocess.run") as mocked_run:
            mocked_run.return_value.stdout = self.mock_output
            commits = get_git_commits()
            self.assertEqual(len(commits), 3)
            for commit in commits:
                self.assertIn(commit, self.mock_output.splitlines())

    def test_parse_commit_line(self) -> None:
        # Test to verify parse_commit_line function
        for timestamp, message in self.parsed_commits:
            result = parse_commit_line(f"{int(timestamp.timestamp())} {message}")
            self.assertEqual(result[0], timestamp)
            self.assertEqual(result[1], message)

    def test_mark_intervals(self) -> None:
        activity = mark_intervals(self.parsed_commits)  # Use self.parsed_commits directly

        expected_activity = {
            datetime(2021, 4, 1, tzinfo=UTC).date(): [0] * 17 + [1] + [0] * 30 + [1] + [0] * 48,
            datetime(2021, 5, 1, tzinfo=UTC).date(): [0] * 48,
        }

        for day, expected_values in expected_activity.items():
            self.assertIn(day, activity)
            self.assertEqual(activity[day], expected_values)

    def test_calculate_daily_hours(self) -> None:
        # Test to verify calculate_daily_hours function
        sample_activity = {
            datetime(2021, 4, 1, tzinfo=UTC).date(): [0] * 17 + [1] + [0] * 30 + [1] + [0] * 48,
            datetime(2021, 5, 1, tzinfo=UTC).date(): [0] * 48,
        }

        daily_hours = calculate_daily_hours(sample_activity)
        self.assertEqual(
            daily_hours[datetime(2021, 4, 1, tzinfo=UTC).date()],
            1.00,
        )
        self.assertEqual(daily_hours[datetime(2021, 5, 1, tzinfo=UTC).date()], 0.00)
