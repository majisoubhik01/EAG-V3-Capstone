[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Command
)

$hadPreviousToken = Test-Path Env:AGENTSWITCH_TOKEN
$previousToken = $env:AGENTSWITCH_TOKEN
$hasUsablePreviousToken = -not [string]::IsNullOrWhiteSpace([string]$previousToken)
$temporaryToken = $false

try {
    if (-not $hasUsablePreviousToken) {
        if (-not (Get-Command Connect-AgentSwitch -ErrorAction SilentlyContinue)) {
            throw "Connect-AgentSwitch is not available in this PowerShell session."
        }

        Remove-Item Env:AGENTSWITCH_TOKEN -ErrorAction SilentlyContinue

        try {
            Connect-AgentSwitch *> $null
        }
        catch {
            throw "AgentSwitch authentication failed."
        }

        if ([string]::IsNullOrWhiteSpace([string]$env:AGENTSWITCH_TOKEN)) {
            throw "AgentSwitch authentication did not produce a bearer token."
        }
        $temporaryToken = $true
    }

    $global:LASTEXITCODE = $null
    Invoke-Expression $Command
    $commandExitCode = $global:LASTEXITCODE
    if ($null -ne $commandExitCode -and $commandExitCode -ne 0) {
        throw "Command failed with exit code $commandExitCode."
    }
}
finally {
    if ($hadPreviousToken) {
        $env:AGENTSWITCH_TOKEN = $previousToken
    }
    else {
        Remove-Item Env:AGENTSWITCH_TOKEN -ErrorAction SilentlyContinue
    }
}