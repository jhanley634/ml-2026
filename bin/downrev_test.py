import unittest
from pathlib import Path

from packaging.version import Version

from bin.downrev import find_downrev_dependencies


class DownrevTest(unittest.TestCase):

    def setUp(self) -> None:
        self.pyproject_content = """
        [project]
        dependencies = [
            "packageA>=1.0.0",
            "packageB>=2.0.0"
        ]
        """

        self.uv_lock_content = """
        [[package]]
        name = "packageA"
        version = "0.9.0"

        [[package]]
        name = "packageB"
        version = "2.1.0"
        """

        self.pyproject_path = Path("test_pyproject.toml")
        with self.pyproject_path.open("w") as f:
            f.write(self.pyproject_content.strip())

        # Write the uv.lock file
        self.uv_lock_path = Path("test_uv.lock")
        with self.uv_lock_path.open("w") as f:
            f.write(self.uv_lock_content.strip())

    def tearDown(self) -> None:
        # Clean up the test files
        self.pyproject_path.unlink(missing_ok=True)
        self.uv_lock_path.unlink(missing_ok=True)

    def test_find_downrev_dependencies(self) -> None:
        expected_downrev_versions = [Version("0.9.0")]
        downrev_dependencies = find_downrev_dependencies(self.pyproject_path, self.uv_lock_path)

        self.assertEqual(sorted(downrev_dependencies), sorted(expected_downrev_versions))
