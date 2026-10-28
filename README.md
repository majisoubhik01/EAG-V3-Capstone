# EAG V3 Capstone

## Persisted evaluation boundary

`RawRunJournal` writes a versioned JSON artifact containing the complete `RawRun` evidence and an `EvaluationManifest` before scoring. The journal redacts credential-shaped fields and bearer values, and its loader rejects malformed, incomplete, or incompatible artifacts. `DeterministicScorer` consumes a loaded run plus independently supplied predicate state and returns only `approve`, `revise`, or `unevaluated`; unavailable live predicates always produce `unevaluated`, which is never a pass. Scoring does not invoke the model, `CalendarAgent`, MCP, or mutations, so a stored run can be rescored with a new scorer version without rerunning it.
## Seat 19: Scheduling

This repository contains the implementation of Seat 19's Calendar Agent. Its charter is:

> Finding time, and moving everything when a date shifts.

The official goals are `calendar.find_30_minutes` and `calendar.move_after_audit`. Both currently have read-only deterministic planners; neither performs calendar mutations.

## Current Status

The repository contains deterministic read-only goal handlers, an optional
provider-injected runtime, a hosted runner for both supported instances, and
an offline-first Keystone adapter. Keystone business reads remain disabled
for default clients; the hosted runner explicitly opts into only the four
allowlisted resource reads. Neither goal performs calendar mutations, and
`PLANNED` is not equivalent to a completed goal.
The older AgentSwitch path and the hosted harness are separate and can contact
live services when explicitly invoked with credentials. This repository
snapshot does not verify their live behavior.

The project provides:

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

- `calendar.find_30_minutes` can return a structured, state-supported candidate slot when a deterministic Party, owned calendar, linked availability rule, visible free/busy result, minimum-notice window, and 30-minute preference are all available. Hidden/not-shared free/busy is not treated as free. Current live data was not rechecked for this documentation snapshot.
- `calendar.move_after_audit` can return a read-only `PLANNED` rescheduling plan when exactly one audit is resolved, an
  explicit calendar scope and positive shift are supplied, and every `CalendarEvent.list` page is validated as complete;
  it does not move any events.

Goal 1 never creates or books the candidate event. Goal 2 is read-only by
default: `PLANNED` means a proposal is ready, not that events moved. A separate
follow-up can execute a fresh move plan only after the user's reply explicitly
confirms the exact proposed shift and the caller opts into disposable-copy
mutations. The normal hosted result checks remain read-only. Keystone mutation
transport additionally requires an exact, explicitly approved HTTPS disposable
base URL; IP targets and the shared AgentSwitch live domain and its subdomains
are rejected. Official
evaluator predicates/scorer and final mutation semantics remain unavailable.

## Harness demonstration

The Suryodaya Harness lives in `harnesses/suryodaya_harness/` alongside the separate Keystone Harness in `harnesses/keystone_harness/`. The Suryodaya Harness is an independent, deterministic verification layer around the real `CalendarAgent` and `BoundedAgentRuntime`; it does not duplicate goal-handler logic.

Run the local demo:

```powershell
python -m harnesses.suryodaya_harness
```

The demo uses deterministic in-memory Calendar tools and a fixed model decision, then runs the Agent through the Harness, verifies the structured result, writes `.harness-artifacts/demo.json`, reloads that RawRun artifact, and evaluates it. The artifact directory is ignored by Git. No provider, live MCP call, credentials, or mutation is required.

### Phase 1 local scheduling surface

The independent Suryodaya Harness also includes deterministic fixtures for the newer scheduling surface in `harnesses.suryodaya_harness.phase1`. These model paginated Party identity data, attendee roles and responses, timezone-aware and malformed free/busy intervals, scheduling-link duration mismatches, audit buffers and strict event boundaries, booking/event consistency, subscription freshness, workflow side effects, risk-based tool policy, execution evidence, and provider or authority failures.

These fixtures are local test data only. Planning remains distinct from execution and proposals remain distinct from committed mutations. The Harness does not call the live Suryodaya deployment, Keystone, a provider, or an external communication/mutation endpoint; provider HTTP 401 evidence is kept distinct from agent or tool permission failures. Live LLM integration, mutation execution, and authoritative subscription or evaluator data remain intentionally unsupported.

### Hosted AgentSwitch harness

`agentswitch-harness.toml` selects both Suryodaya and Keystone and invokes
`harnesses.suryodaya_harness.hosted_runner`. The hosted process reads
`AGENTSWITCH_INSTANCE`, `AGENTSWITCH_BASE_URL`, and `AGENTSWITCH_TOKEN` from
the runner; it does not read local credentials or `.env` files. The base URL
must be an absolute credential-free HTTP or HTTPS instance base URL. The
client appends `/api/mcp`, so that endpoint itself must not be supplied. Query
strings and fragments are rejected.

The hosted runner includes separate read-only tasks for
`calendar.find_30_minutes` and `calendar.move_after_audit`, each with an
explicit harness-owned predicate. Goal 1 passes only when the planner returns
one structurally valid 30-minute candidate and a fresh harness read confirms
the participant, owned calendar, preferences, linked availability rule, and
free/busy state. Clarifications are reported with allowlisted reason codes,
not raw participant or calendar details. It never books the candidate. For
Suryodaya, Goal 2 resolves
`EVT-2026-00007` by its supported event-number filter, checks its title,
confirmed status, and event-associated calendar, and requires a complete
event listing before planning. Explicit offsets remain instants.
When a timestamp has no event timezone, the instructor-confirmed rule is to
use the owning calendar's timezone, then the company's timezone. The current
tool interface can read the owning calendar timezone but does not expose a
separate company-timezone read; it fails closed if no usable timezone is
returned until the server applies the documented fallback. Ambiguous and
nonexistent local times fail closed. Offset-versus-IANA precedence remains
unverified. The
30-minute proposal is presented for confirmation or change; events are never
updated. The harness-owned Goal 2 predicate passes only for a unique confirmed
audit, a complete conflict-free plan, and a non-empty confirmation request.
The planner first proposes 30 minutes; if that conflicts, the hosted check may
search 30-minute increments up to eight hours and present the first
conflict-free alternative. Every proposal still requires confirmation, and no
event is moved by this check. Ambiguous audits, conflicts throughout the
bounded search, or incomplete evidence remain non-passing.
Before each hosted predicate scores an executed task, the runner persists a
redacted execution journal under
`.harness-artifacts/runs/<run_id>/<instance>/`. The journal omits tool
arguments, records structured result facts and tool-call metadata, then stores
the independent check result. `results.json` remains the platform-facing
summary.
The normal hosted result runner never applies a proposal. A separate
`CalendarAgent.confirm_move` follow-up requires an affirmative message that
repeats the proposed duration, a fresh conflict check, and explicit
disposable-write opt-ins on both the agent and tool adapter. Keystone also
requires its hosted mutation transport to be explicitly enabled and the
`CalendarEvent.update` descriptor to be discovered, with the exact disposable
base URL explicitly approved by the caller. These write capabilities are not
enabled by the standard `agentswitch-harness.toml` command.

For Keystone, the hosted-only client requires an observed, schema-validated
`tools/list` descriptor for each read tool before making one-shot `tools/call` requests for
`Party.list`, `Calendar.list`, `CalendarEvent.list`, or `EventAttendee.list`.
Goal 1 additionally uses the exact observed read-only descriptors for
`SchedulingPreferences.list`, `AvailabilityRule.list`, and
`endpoint.calendar.free_busy`. These operations have a separate narrow
allowlist, and only `structuredContent` matching each discovered output schema
is used; text-only results fail closed. A local tools/list capture found all
three descriptors, but its catalog completeness was unknown and it did not
read business data. A later harness check must independently establish unique
identity, preferences, a relevant availability rule, and visible free/busy
before Goal 1 can pass. Goal 2 requires exactly one audit search result and its
owning calendar before producing a 30-minute plan for confirmation. These are
harness-owned checks, not official evaluator results. `results.json` reports
the checks and the runner summary explicitly says `official_evaluator=not_invoked`.
The hosted command exits successfully after writing a valid result document,
even when individual checks fail; task outcomes are reported in each result
record. A nonzero exit is reserved for runner or result-generation errors.

The earlier live Suryodaya snapshot could not produce a Goal 2 plan under the
then-strict timestamp parser because event timestamps lacked offsets and most
records had no event timezone. The explicit fallback can now support a
read-only proposal, but that snapshot was not re-read and pagination snapshot
consistency remains unverified. The Goal 1 snapshot also lacked matching
availability rules. A fresh read-only check on 2026-10-28 found one owned
calendar and one 30-minute preference record, but no rule linked by party or
calendar; the named company-level “Plant visiting hours” rule is not linked
to either. The free/busy response returned `freebusy.notShared` with access
`none`, so availability remains unverified and is not a goal pass.
The hosted runner does not mutate events, and the local
`python -m harnesses.suryodaya_harness` demo remains fixture-based and does not
contact a live instance.

The interactive read-only checks used the instructor-provided
`https://agentswitch.theschoolofai.in` endpoint. Hosted execution must use
the platform-injected instance URL and credentials; do not hardcode or infer
credentials from local files.

Replay and rescore an existing artifact without rerunning the Agent:

```powershell
python -m harnesses.suryodaya_harness replay .harness-artifacts/demo.json
```

Replay loads the persisted artifact and invokes only the deterministic scorer. It does not invoke the model, `CalendarAgent`, Calendar tools, MCP, or mutations. With no externally supplied predicate, the result is `UNEVALUATED`, which is not approval. An explicit predicate can be supplied when independently available:

```powershell
python -m harnesses.suryodaya_harness replay .harness-artifacts/demo.json --predicate passed
```

Current evaluation is deterministic and predicate-state driven. The demo supplies its independent Harness verification as the predicate; replay does not infer evaluation from model decisions, prose, or `claimed_outcome`.

## Authentication

The client reads credentials from the local environment. Credentials must not be committed to Git.

For bearer-token authentication, set:

```powershell
$env:AGENTSWITCH_TOKEN = "your-local-token"
```

The instance base URL must also be supplied explicitly by the authorized
runtime. `AGENTSWITCH_BASE_URL` has no public default; the client appends
`/api/mcp` to the supplied instance base URL.
For local use, set it in the PowerShell session to the authorized HTTPS
instance base URL before invoking `scripts/run.ps1`; that helper bootstraps
the token but does not choose or supply the instance URL.

Optional setting:

```powershell
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

The workspace's VS Code settings enable pytest discovery under `tests`. Use
VS Code `#runTests` and confirm that the intended test nodes were collected
when the task requires Testing-panel evidence; an aggregate result is not
proof that a newly added test was included.

Run the read-only live smoke test through the persistent local authentication bootstrap:

```powershell
.\scripts\run.ps1 -Command "python scripts/smoke_test.py"
```

The smoke test calls `CalendarEvent.list` with `limit=1`, validates the MCP/JSON-RPC response, and prints only safe metadata. It talks to the live AgentSwitch environment and must not be used with credentials you do not intend to use there.

LLM integration, evaluator/scorer integration, mutation execution, and successful mutation-based completion of both official Calendar goals remain intentionally unimplemented. The live environment does not expose the Seat 19 evaluator predicates, so this repository does not invent them.

## Engineering documentation

For a component map, start with
[Engineering Architecture](docs/engineering/Architecture.md). The engineering
documentation also records execution flows, decisions, safety constraints,
bug/feature status, test evidence, recovery guidance, the current handover,
and a progressive learning path:

- [Architecture](docs/engineering/Architecture.md)
- [Execution flows](docs/engineering/Flow.md)
- [Decision log](docs/engineering/Decisions.md)
- [Constraints](docs/engineering/Constraints.md)
- [Bug register](docs/engineering/Bug.md)
- [Feature register](docs/engineering/Feature.md)
- [Test checklist](docs/engineering/Test-Checklist.md)
- [Rollback guide](docs/engineering/Rollback.md)
- [AI work log](docs/engineering/AI-Work-Log.md)
- [Learning guide](docs/engineering/LEARNING-GUIDE.md)
- [Current handover](docs/engineering/HANDOVER.md)
