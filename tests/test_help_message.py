"""Tests the help message."""

from .utils import HYDRA_CLI_INCOMPATIBLE_MARKER, hydra_cli_unavailable, run_command


def test_no_argument_usage():
    """The no-argument path returns before hydra is touched, so it behaves the same everywhere."""

    _usage_stderr, usage_stdout = run_command("aces-cli", {}, "usage", expected_returncode=1)
    assert "Usage: aces-cli [OPTIONS]" in usage_stdout, (
        f"Expected usage message not found in stdout. Got {usage_stdout}"
    )


def test_e2e():
    """`aces-cli -h` prints hydra's help, or explains why hydra cannot start."""

    if hydra_cli_unavailable():
        help_stderr, _help_stdout = run_command("aces-cli -h", {}, "help", expected_returncode=1)
        assert HYDRA_CLI_INCOMPATIBLE_MARKER in help_stderr, (
            f"Expected the hydra incompatibility message on stderr. Got {help_stderr}"
        )
        assert "Traceback (most recent call last)" not in help_stderr, (
            f"A traceback leaked instead of the guard message. Got {help_stderr}"
        )
        return

    _help_stderr, help_stdout = run_command("aces-cli -h", {}, "help", expected_returncode=0)
    assert "== aces-cli ==" in help_stdout, f"Expected help message not found in stdout. Got {help_stdout}"


def test_hydra_failure_is_argument_independent():
    """Hydra builds its parser before reading argv, so real arguments fail exactly as `-h` does.

    This is what justifies the guard's "cannot start" wording: the CLI is unusable regardless of how it is
    invoked, not merely broken in help mode.
    """

    if not hydra_cli_unavailable():
        _stderr, _stdout = run_command(
            "aces-cli --help", {}, "args are independent of the failure", expected_returncode=0
        )
        return

    stderr, _stdout = run_command(
        "aces-cli cohort_name=x cohort_dir=/tmp/aces-cli-probe",
        {},
        "hydra fails before reading argv",
        expected_returncode=1,
    )
    assert HYDRA_CLI_INCOMPATIBLE_MARKER in stderr, (
        f"Expected the hydra incompatibility message on stderr. Got {stderr}"
    )
