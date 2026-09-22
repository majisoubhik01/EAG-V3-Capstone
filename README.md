# EAG V3 Capstone

## Seat 19: Scheduling

This repository contains the implementation of Seat 19's Calendar Agent. Its charter is:

> Finding time, and moving everything when a date shifts.

The official goals are `calendar.find_30_minutes` and `calendar.move_after_audit`. Both currently have read-only deterministic planners; neither performs calendar mutations.

## Current Status

The read-only AgentSwitch integration and the first planning/harness architecture pass are available. The current agent does not execute mutations or claim either goal is complete. It provides:

- environment-backed authentication configuration
- a reusable JSON-RPC 2.0 MCP client
- a read-only `CalendarEvent.list` wrapper
- an explicit read-only Calendar tool boundary
- ambiguity-first handlers for both Calendar goals
- an independent, fake-tool harness for handwritten tests
- unit tests for request creation, response/error handling, planning, and verification
- an explicitly marked live smoke test script

No CalendarEvents are created, updated, or deleted, and no bookings are modified.

## Architecture

```text
AgentTask
	-> CalendarAgent
		-> goal handler / planner
			-> CalendarTool interface
				-> AgentSwitchCalendarTools
					-> AgentSwitchClient
						-> JSON-RPC MCP
```

The `CalendarAgent` owns orchestration and structured results. Goal handlers own domain decisions and stop with explicit clarification when live data is ambiguous. `CalendarTool` owns the read-side operation boundary and records tool calls; it does not expose raw HTTP details to the agent. The harness runs the agent against fake in-memory tools and independently checks outcome status and mutation-free integrity. It does not treat an agent's success prose as proof.

The handlers remain read-only and stop before event execution:

- `calendar.find_30_minutes` can return a structured, state-supported candidate slot when a deterministic Party, owned calendar, linked availability rule, visible free/busy result, minimum-notice window, and 30-minute preference are all available. It cannot identify a unique Plant Manager from current live Party data, and hidden/not-shared free/busy is not treated as free.
- `calendar.move_after_audit` can return a read-only `PLANNED` rescheduling plan when exactly one audit is resolved and an explicit positive shift is supplied; it does not move any events.

Goal 1 never creates or books the candidate event, and Goal 2 never updates events. `PLANNED` means only that the read-only state supports a candidate slot or rescheduling plan; it is not equivalent to booking or moving events. The official evaluator predicates/scorer and final mutation semantics remain unavailable and unknown.

## Authentication

The client reads credentials from the local environment. Credentials must not be committed to Git.

For bearer-token authentication, set:

```powershell
$env:AGENTSWITCH_TOKEN = "your-local-token"
```

Optional settings:

```powershell
$env:AGENTSWITCH_BASE_URL = "https://agentswitch.theschoolofai.in"
$env:AGENTSWITCH_TIMEOUT_SECONDS = "10"
```

The client expects a bearer token obtained through the verified AgentSwitch login flow and sends it as `Authorization: Bearer <token>` to the default `POST /api/mcp` endpoint. Browser cookies, CSRF handling, and the login request itself are intentionally outside this client until verified. Do not copy `team19.auth.json` into this repository.

## Local AgentSwitch authentication

`Connect-AgentSwitch` is provided by a shared user-local PowerShell module at `$HOME\Documents\PowerShell\Modules\AgentSwitchAuth\AgentSwitchAuth.psm1`. Both the Windows PowerShell 5.1 and PowerShell 7 profiles import that same module. On first use, the command prompts with `Get-Credential`; enter the AgentSwitch email and password when prompted. The password is held in a `PSCredential` only for the login request and is not written to disk.

After login, the module stores the bearer token outside the repository at `%LOCALAPPDATA%\AgentSwitch\token.dpapi`. The file contains a `ConvertFrom-SecureString` DPAPI payload tied to the current Windows user, not plaintext. A later `Connect-AgentSwitch` decrypts that cache locally and avoids prompting. If a read-only MCP request receives HTTP 401, the module removes the cached token, prompts once for fresh credentials, caches the new token, and retries that read-only request exactly once. Token lifetime is not inferred from token contents.

The repository includes `scripts/run.ps1` for local commands that need AgentSwitch access. It reuses an existing process token or invokes `Connect-AgentSwitch` to load the DPAPI cache, passes the token to Python through the process-only `AGENTSWITCH_TOKEN` environment variable, and removes or restores the variable when the command finishes. Never commit tokens, passwords, cookies, auth state, or browser sessions.

Examples:

```powershell
.\scripts\run.ps1 -Command "python -m pytest"
.\scripts\run.ps1 -Command "python scripts/smoke_test.py"
```

The command is intentionally required because the final Calendar Agent entrypoint has not been selected yet. The script requires `Connect-AgentSwitch` to be available in the current PowerShell session and does not implement browser-cookie authentication. The shared module can also be imported manually from either PowerShell profile; `Connect-AgentSwitch` sets the process-only bearer token for the existing `MCP-Call` and `MCP-Request` helper commands.

## Tests

The unit tests do not require AgentSwitch credentials or network access:

```powershell
python -m pytest
```

Run the read-only live smoke test through the persistent local authentication bootstrap:

```powershell
.\scripts\run.ps1 -Command "python scripts/smoke_test.py"
```

The smoke test calls `CalendarEvent.list` with `limit=1`, validates the MCP/JSON-RPC response, and prints only safe metadata. It talks to the live AgentSwitch environment and must not be used with credentials you do not intend to use there.

LLM integration, evaluator/scorer integration, mutation execution, and successful mutation-based completion of both official Calendar goals remain intentionally unimplemented. The live environment does not expose the Seat 19 evaluator predicates, so this repository does not invent them.