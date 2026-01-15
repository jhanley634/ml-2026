import unittest
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

from git.log_analysis import find_daily_counts, get_git_commits, mark_intervals, parse_commit_line

UTC = ZoneInfo("UTC")


class TestGitLogAnalysis(unittest.TestCase):

    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        self.parsed_commits = [(datetime.now(tz=UTC), "")]

    def setUp(self) -> None:
        self.parsed_commits = [
            (datetime(2021, 4, 1, 9, tzinfo=UTC), "Commit message 1"),
            (datetime(2021, 4, 1, 10, tzinfo=UTC), "Commit message 2"),
            (datetime(2021, 5, 1, 0, tzinfo=UTC), "Commit message 3"),
        ]

    def test_get_git_commits(self) -> None:
        mock_output = """\
            1617187200 Commit message 1
            1617190800 Commit message 2
            1617210000 Commit message 3"""

        with mock.patch("subprocess.run") as mocked_run:
            mocked_run.return_value.stdout = mock_output
            commits = get_git_commits()

            # Assert subprocess is invoked with the expected git command and flags
            mocked_run.assert_called_once()
            called_args, called_kwargs = mocked_run.call_args
            expected_command = ["git", "reflog", "--pretty=format:%at %gs"]
            self.assertEqual(called_args[0], expected_command)
            self.assertTrue(called_kwargs["check"])
            self.assertTrue(called_kwargs["capture_output"])
            self.assertTrue(called_kwargs["text"])

            self.assertEqual(len(commits), 3)
            for commit in commits:
                self.assertIn(commit, mock_output.splitlines())

    def test_parse_commit_line(self) -> None:
        for timestamp, message in self.parsed_commits:
            result = parse_commit_line(f"{int(timestamp.timestamp())} {message}")
            self.assertEqual(result[0], timestamp)
            self.assertEqual(result[1], message)

    def test_mark_intervals(self) -> None:
        # Add two commits that fall into the same half-hour interval as the existing 09:00 commit
        extra_commits = [
            (
                datetime(2021, 4, 1, hour=9, minute=5, tzinfo=UTC),
                "extra commit in 09:00-09:30 interval A",
            ),
            (
                datetime(2021, 4, 1, hour=9, minute=25, tzinfo=UTC),
                "extra commit in 09:00-09:30 interval B",
            ),
        ]
        activity = mark_intervals(self.parsed_commits + extra_commits)

        expected_activity = {
            # Original 09:00 bucket plus two extra commits in the same 30-minute interval
            datetime(2021, 4, 1, hour=9, tzinfo=UTC): 3,
            datetime(2021, 4, 1, hour=10, tzinfo=UTC): 1,
            datetime(2021, 5, 1, hour=0, tzinfo=UTC): 1,
        }
        self.assertEqual(set(expected_activity.keys()), set(activity.keys()))
        for bucket in activity:
            self.assertEqual(expected_activity[bucket], activity[bucket])

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
                "stamp": expected_data.keys(),
                "count": expected_data.values(),
            },
        ).set_index("stamp")

        result_df = find_daily_counts(sample_activity)

        pd.testing.assert_frame_equal(expected_df, result_df)
