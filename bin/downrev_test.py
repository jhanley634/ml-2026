import unittest
from pathlib import Path

from packaging.version import Version

from bin.downrev import find_downrev_dependencies

temp = Path("/tmp")


class DownrevTest(unittest.TestCase):

    pyproject_content = ""
    uv_lock_content = ""

    uv_lock_path = pyproject_path = Path("/tmp")

    def setUp(self) -> None:
        self.pyproject_content = """
        [project]
        dependencies = [
            "numpy >= 2.3.3",
            "requests >= 2.99.0",
        ]
        """

        self.uv_lock_content = """
        [[package]]
        name = "numpy"
        version = "2.4.5"

        [[package]]
        name = "requests"
        version = "2.33.1"
        """

        self.pyproject_path = temp / "test_pyproject.toml"
        with self.pyproject_path.open("w") as f:
            f.write(self.pyproject_content.strip())

        # Write the uv.lock file
        self.uv_lock_path = temp / "test_uv.lock"
        with self.uv_lock_path.open("w") as f:
            f.write(self.uv_lock_content.strip())

    def tearDown(self) -> None:
        self.pyproject_path.unlink(missing_ok=True)
        self.uv_lock_path.unlink(missing_ok=True)

    def test_find_downrev_dependencies(self) -> None:
        expected_downrev_versions = [Version("2.33.1")]
        downrev_dependencies = find_downrev_dependencies(self.pyproject_path, self.uv_lock_path)

        self.assertEqual(
            sorted(downrev_dependencies),
            sorted(expected_downrev_versions),
        )
