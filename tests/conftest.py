"""Test-only database isolation before application modules are imported."""

import os
from pathlib import Path


_test_database = Path("tests") / ".tmp_test_suite.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_test_database.as_posix()}"
