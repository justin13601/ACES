"""Tests the help message."""

from .utils import run_command


def test_e2e():
    # Running with the empty directory
    help_stderr, help_stdout = run_command("aces-cli", {}, "help", expected_returncode=1)
    assert "Usage: aces-cli [OPTIONS]" in help_stdout, (
        f"Expected help message not found in stdout. Got {help_stdout}"
    )

    # Running with the empty directory
    help_stderr, help_stdout = run_command("aces-cli -h", {}, "help", expected_returncode=0)
    assert "== aces-cli ==" in help_stdout, f"Expected help message not found in stdout. Got {help_stdout}"
