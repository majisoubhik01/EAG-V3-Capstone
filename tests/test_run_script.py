from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


pytestmark = pytest.mark.skipif(
    POWERSHELL is None, reason="PowerShell is required for bootstrap tests"
)


def run_powershell(
    command: str, token: str | None = None
) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None
    environment = {
        key: value for key, value in os.environ.items() if key != "AGENTSWITCH_TOKEN"
    }
    if token is not None:
        environment["AGENTSWITCH_TOKEN"] = token
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_script_passes_fake_token_to_child_without_printing_it() -> None:
    command = (
        "function Connect-AgentSwitch { $env:AGENTSWITCH_TOKEN = 'fake-token-for-test' }; "
        f"& '{RUN_SCRIPT}' -Command 'python -c \"import os; "
        "raise SystemExit(os.getenv(''AGENTSWITCH_TOKEN'') != ''fake-token-for-test'')\"'; "
        "if (Test-Path Env:AGENTSWITCH_TOKEN) { exit 1 }"
    )

    result = run_powershell(command)

    assert result.returncode == 0, result.stderr
    assert "fake-token-for-test" not in result.stdout
    assert "fake-token-for-test" not in result.stderr


def test_run_script_reuses_existing_token_without_authentication() -> None:
    command = (
        "function Connect-AgentSwitch { throw 'authentication should not run' }; "
        f"& '{RUN_SCRIPT}' -Command 'python -c \"import os; "
        "raise SystemExit(os.getenv(''AGENTSWITCH_TOKEN'') != ''parent-token'')\"'"
    )

    result = run_powershell(command, token="parent-token")

    assert result.returncode == 0, result.stderr
    assert "authentication should not run" not in result.stderr


def test_run_script_authenticates_when_token_is_missing() -> None:
    command = (
        "function Connect-AgentSwitch { $env:AGENTSWITCH_TOKEN = 'fresh-token' }; "
        f"& '{RUN_SCRIPT}' -Command 'python -c \"import os; "
        "raise SystemExit(os.getenv(''AGENTSWITCH_TOKEN'') != ''fresh-token'')\"'; "
        "if (Test-Path Env:AGENTSWITCH_TOKEN) { exit 1 }"
    )

    result = run_powershell(command)

    assert result.returncode == 0, result.stderr


def test_run_script_allows_successful_powershell_command() -> None:
    result = run_powershell(
        f"& '{RUN_SCRIPT}' -Command 'Write-Output authenticated-child'",
        token="parent-token",
    )

    assert result.returncode == 0, result.stderr
    assert "authenticated-child" in result.stdout


def test_run_script_rejects_nonzero_child_exit_code() -> None:
    result = run_powershell(
        f"& '{RUN_SCRIPT}' -Command 'python -c \"raise SystemExit(7)\"'",
        token="parent-token",
    )

    assert result.returncode != 0
    assert "exit code 7" in result.stderr


def test_run_script_preserves_existing_parent_token() -> None:
    command = (
        f"& '{RUN_SCRIPT}' -Command 'Write-Output child'; "
        "if ($env:AGENTSWITCH_TOKEN -ne 'parent-token') { exit 1 }"
    )

    result = run_powershell(command, token="parent-token")

    assert result.returncode == 0, result.stderr


def test_run_script_cleans_up_temporary_token() -> None:
    command = (
        "function Connect-AgentSwitch { $env:AGENTSWITCH_TOKEN = 'temporary-token' }; "
        f"& '{RUN_SCRIPT}' -Command 'Write-Output child'; "
        "if (Test-Path Env:AGENTSWITCH_TOKEN) { exit 1 }"
    )

    result = run_powershell(command)

    assert result.returncode == 0, result.stderr


def test_run_script_reports_missing_auth_command() -> None:
    result = run_powershell(f"& '{RUN_SCRIPT}' -Command \"python -m pytest\"")

    assert result.returncode != 0
    assert "Connect-AgentSwitch is not available" in result.stderr