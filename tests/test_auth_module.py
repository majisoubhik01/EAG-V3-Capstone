from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run.ps1"
MODULE_PATH = (
    Path.home()
    / "Documents"
    / "PowerShell"
    / "Modules"
    / "AgentSwitchAuth"
    / "AgentSwitchAuth.psm1"
)
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


pytestmark = pytest.mark.skipif(
    POWERSHELL is None or os.name != "nt" or not MODULE_PATH.exists(),
    reason="Windows PowerShell and the local AgentSwitchAuth module are required",
)


def run_powershell(script: str) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"AGENTSWITCH_TOKEN", "AGENTSWITCH_TOKEN_CACHE_PATH"}
    }
    return subprocess.run(
        [
            POWERSHELL,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def auth_setup() -> str:
    return f"""
$ErrorActionPreference = 'Stop'
$cache = Join-Path ([IO.Path]::GetTempPath()) ('agentswitch-test-' + [Guid]::NewGuid().ToString() + '.dpapi')
$env:AGENTSWITCH_TOKEN_CACHE_PATH = $cache
Remove-Item Env:AGENTSWITCH_TOKEN -ErrorAction SilentlyContinue
$modulePath = '{MODULE_PATH}'
"""


def auth_mocks(
    login_body: str = "return [pscustomobject]@{ token = 'fresh-token' }",
    mcp_body: str = "return [pscustomobject]@{ jsonrpc = '2.0'; id = 1; result = @{ isError = $false } }",
) -> str:
    return f"""
function Get-Credential {{
    $script:promptCalls++
    return New-Object System.Management.Automation.PSCredential(
        'test@example.invalid',
        (ConvertTo-SecureString 'fake-password' -AsPlainText -Force)
    )
}}
function Invoke-RestMethod {{
    [CmdletBinding()]
    param([string]$Uri, [string]$Method, [string]$ContentType, [string]$Body, [hashtable]$Headers)
    if ($Uri -like '*/api/auth/login') {{
        $script:loginCalls++
        {login_body}
    }}
    $script:mcpCalls++
    {mcp_body}
}}
"""


def cleanup_script() -> str:
    return """
Remove-Item $cache -Force -ErrorAction SilentlyContinue
Remove-Item Env:AGENTSWITCH_TOKEN_CACHE_PATH -ErrorAction SilentlyContinue
Remove-Item Env:AGENTSWITCH_TOKEN -ErrorAction SilentlyContinue
exit 0
"""


def test_no_cached_token_uses_credential_login() -> None:
    script = (
        auth_setup()
        + "$script:loginCalls = 0; $script:promptCalls = 0; $script:mcpCalls = 0\n"
        + auth_mocks()
        + f"Import-Module '{MODULE_PATH}' -Force -DisableNameChecking\n"
        + "Connect-AgentSwitch\n"
        + "if ($script:loginCalls -ne 1 -or $script:promptCalls -ne 1) { exit 1 }\n"
        + "Write-Output 'no-cache-login-passed'\n"
        + cleanup_script()
    )

    result = run_powershell(script)

    assert result.returncode == 0, result.stderr
    assert "no-cache-login-passed" in result.stdout


def test_cached_token_skips_credential_prompt_and_is_not_plaintext() -> None:
    script = (
        auth_setup()
        + "$script:loginCalls = 0; $script:promptCalls = 0; $script:mcpCalls = 0\n"
        + auth_mocks()
        + f"Import-Module '{MODULE_PATH}' -Force -DisableNameChecking\n"
        + "Connect-AgentSwitch\n"
        + "Remove-Item Env:AGENTSWITCH_TOKEN\n"
        + "function Get-Credential { throw 'prompt should not run' }\n"
        + "Connect-AgentSwitch\n"
        + "$protected = Get-Content $cache -Raw\n"
        + "if ($script:loginCalls -ne 1 -or $protected -like '*fresh-token*') { exit 1 }\n"
        + "Write-Output 'cache-reuse-passed'\n"
        + cleanup_script()
    )

    result = run_powershell(script)

    assert result.returncode == 0, result.stderr
    assert "cache-reuse-passed" in result.stdout
    assert "fresh-token" not in result.stdout
    assert "fresh-token" not in result.stderr


def test_cached_token_supports_successful_read_only_mcp_call() -> None:
    script = (
        auth_setup()
        + "$script:loginCalls = 0; $script:promptCalls = 0; $script:mcpCalls = 0\n"
        + auth_mocks()
        + f"Import-Module '{MODULE_PATH}' -Force -DisableNameChecking\n"
        + "Connect-AgentSwitch\n"
        + "Remove-Item Env:AGENTSWITCH_TOKEN\n"
        + "MCP-Call -Tool 'CalendarEvent.list' -Arguments @{ limit = 1 } | Out-Null\n"
        + "if ($script:loginCalls -ne 1 -or $script:mcpCalls -ne 1) { exit 1 }\n"
        + "Write-Output 'cached-mcp-passed'\n"
        + cleanup_script()
    )

    result = run_powershell(script)

    assert result.returncode == 0, result.stderr
    assert "cached-mcp-passed" in result.stdout


def test_cached_token_401_reauthenticates_and_retries_once() -> None:
    retry_mock = """
if ($Uri -like '*/api/auth/login') {
    $script:loginCalls++
    return [pscustomobject]@{ token = ('token-' + $script:loginCalls) }
}
if ($script:mcpCalls -eq 1) {
    $exception = [Exception]::new('unauthorized')
    $exception | Add-Member -MemberType NoteProperty -Name Response -Value ([pscustomobject]@{ StatusCode = 401 }) -Force
    throw $exception
}
return [pscustomobject]@{ jsonrpc = '2.0'; id = 1; result = @{ isError = $false } }
"""
    script = (
        auth_setup()
        + "$script:loginCalls = 0; $script:promptCalls = 0; $script:mcpCalls = 0\n"
        + auth_mocks(mcp_body=retry_mock)
        + f"Import-Module '{MODULE_PATH}' -Force -DisableNameChecking\n"
        + "Connect-AgentSwitch\n"
        + "Remove-Item Env:AGENTSWITCH_TOKEN\n"
        + "MCP-Call -Tool 'CalendarEvent.list' -Arguments @{ limit = 1 } | Out-Null\n"
        + "if ($script:loginCalls -ne 2 -or $script:mcpCalls -ne 2 -or $script:promptCalls -ne 2) { exit 1 }\n"
        + "Write-Output '401-retry-passed'\n"
        + cleanup_script()
    )

    result = run_powershell(script)

    assert result.returncode == 0, result.stderr
    assert "401-retry-passed" in result.stdout


def test_reauthentication_failure_is_sanitized() -> None:
    failing_login = "if ($script:failLogin) { throw [Exception]::new('login failed with secret') }; return [pscustomobject]@{ token = 'fresh-token' }"
    unauthorized_mcp = "if ($Uri -like '*/api/auth/login') { $script:loginCalls++; return [pscustomobject]@{ token = 'fresh-token' } }; $exception = [Exception]::new('unauthorized'); $exception | Add-Member -MemberType NoteProperty -Name Response -Value ([pscustomobject]@{ StatusCode = 401 }) -Force; throw $exception"
    script = (
        auth_setup()
        + "$script:loginCalls = 0; $script:promptCalls = 0; $script:mcpCalls = 0\n"
        + auth_mocks(login_body=failing_login, mcp_body=unauthorized_mcp)
        + f"Import-Module '{MODULE_PATH}' -Force -DisableNameChecking\n"
        + "Connect-AgentSwitch\n"
        + "$script:failLogin = $true\n"
        + "Remove-Item Env:AGENTSWITCH_TOKEN\n"
        + "try { MCP-Call -Tool 'CalendarEvent.list' -Arguments @{ limit = 1 } | Out-Null; exit 1 } catch { if ($_.Exception.Message -ne 'AgentSwitch reauthentication failed.') { exit 1 } }\n"
        + "Write-Output 'reauth-failure-passed'\n"
        + cleanup_script()
    )

    result = run_powershell(script)

    assert result.returncode == 0, result.stderr
    assert "reauth-failure-passed" in result.stdout
    assert "secret" not in result.stdout
    assert "secret" not in result.stderr


def test_run_script_uses_cached_token_without_prompting() -> None:
    script = (
        auth_setup()
        + "$script:loginCalls = 0; $script:promptCalls = 0; $script:mcpCalls = 0\n"
        + auth_mocks()
        + f"Import-Module '{MODULE_PATH}' -Force -DisableNameChecking\n"
        + "Connect-AgentSwitch\n"
        + "Remove-Item Env:AGENTSWITCH_TOKEN\n"
        + "function Get-Credential { throw 'prompt should not run' }\n"
        + f"& '{RUN_SCRIPT}' -Command 'if ([string]::IsNullOrWhiteSpace($env:AGENTSWITCH_TOKEN)) {{ throw ''child token missing'' }}; Write-Output cached-child'\n"
        + "if (Test-Path Env:AGENTSWITCH_TOKEN) { exit 1 }\n"
        + "Write-Output 'run-cache-passed'\n"
        + cleanup_script()
    )

    result = run_powershell(script)

    assert result.returncode == 0, result.stderr
    assert "run-cache-passed" in result.stdout