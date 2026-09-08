# Agent Mode (classifier-driven tool approval)

The permission mode where a LOCAL language model judges each tool call the
static ladder would have asked the user about. Mode case `.agentAuto`,
labelled "Agent (classifier)". The shape is qwen-code's Auto Mode; the
difference is the classifier: qwen-code calls a cloud fast model, this app
runs the loaded model through the same `TurboSparkSession.generate` seam the
compaction summarizer uses, so the feature is offline and private by
construction.

Read this page before touching the routing, the counters, the classifier
prompt, or the `hardGated` marking.

**The one-sentence rule: the classifier lives ONLY in the ask-band.** Deny
rules, the terminal denylist, structural shell guards, sensitive paths, and
the repo-import MCP gate keep their human no matter what the classifier
would say; session grants and MCP allow rules keep their silent run; and
everything else the engine scored `.ask` is what the classifier judges.

## Why the existing command classifier cannot do this job

`docs/PERMISSION_GATE.md` is the measured answer: `CommandGate` (the
`permission-gate-onnx` logistic regressions) adds zero true positives
against `TerminalRiskGateTests`' contract lists, false-positives on the
must-run list, and holds out at ROC AUC 0.701 by generator. It is also the
wrong shape: a hazard denylist can only ADD friction, and Agent mode's whole
point is granting auto-approvals. The role split is therefore fixed:

- `CommandGate.advisoryReason` decorates ask verdicts (shipped on).
- `CommandGate.veto` can only push an allowlisted command to ask (shipped
  off, `MacAppSettings.commandAdvisoryVeto`).
- Agent mode's verdicts come from the loaded model, which sees the user's
  recent request and can say both yes and no.

## The routing

`AgentModeRouting.preClassifierDecision` is the shared static both call
sites run; do not re-derive it inline. In order:

| Input (engine returned `.ask`, no hook verdict) | Decision |
|---|---|
| `assessment.isHardGated` | manual card / subagent refusal |
| session suspended, or skip thresholds tripped | manual card, with the skip notice |
| category `.mcp` | classify |
| risk `.safe` or `.low` (non-MCP) | run, no card, no classifier call |
| otherwise (e.g. `.high` "not a recognized command") | classify |

MCP calls always classify because a tool name plus the server's OWN
annotations is exactly what a classifier exists to weigh (qwen-code's rule;
the annotations are forwarded flagged unverified). The `.safe`/`.low` fast
path is the same population `.auto` already runs, so a classifier call
there would only add latency.

The verdicts map: allow -> run (tool card marked "Classifier Approved");
block -> refusal carrying "Blocked by Agent mode policy: <reason>" plus the
don't-route-around guidance line, and `PermissionDenied` fires for hooks
(the batch path marks these `fromEngine: true`, which on that tuple means
"dispatch the hook"); unavailable -> the ordinary card with a fallback
notice.

## Hard gates

`ToolRiskAssessment.hardGated` (optional Bool, nil = soft) marks verdicts
that come from a DETERMINISTIC guard: the denylist regexes and substring
checks, the `isSingleSimpleInvocation` structural failure, the
`CommandGate` veto, sensitive-path reads and writes, malformed/private/
sandbox-denied web targets. An "unrecognized command" ask is deliberately
SOFT -- that population is the feature's reason to exist.

Two invariants worth re-stating:

- The engine's step 7b (repo-imported MCP server) fires in `.auto` AND
  `.agentAuto`, and marks its ask hard. Remove either half and a cloned
  `.mcp.json`'s tools are judged by a model instead of by the user.
- A `PermissionRequest` hook that asks parks the card BEFORE the engine
  runs, so hook asks never classify; hooks are explicit user automation and
  outrank a heuristic.

## The classifier contract

One call, greedy (`temperature 0`, `topK 1`, `topP 1.0`, `maxNewTokens` 96,
reasoning off), 20 s timeout.

- System prompt: role, the output contract, the built-in policy, then the
  user's hints, then environment lines, then the workspace root. This text
  is a STABLE PREFIX across classifications on purpose -- prefix reuse
  makes the second and later classifications in a session cheaper.
- User message: the recent user request (last 2 user messages, 1,500 chars,
  tail-kept) and the projected call.
- Projection (`ToolCallProjection`): the shell command in full; write paths
  plus 300-char content previews; the web URL or query and NOT the prompt
  field; MCP server, tool, annotations, and bounded arguments (2,000
  chars per string, 16,000 shared, cuts marked in place); subagent type
  plus full prompt (the steering surface). Tool results never cross.
- Response: exactly one JSON object -- `{"verdict":"allow"}` or
  `{"verdict":"block","reason":"..."}`. Parsing tolerates fences and prose
  (the same measured hazard SKILL.state handled); an unknown verdict string
  is NOT a verdict. One corrective retry, then `.unavailable`.
- No constrained decoding: `docs/SKILL_STATE.md` measured post-hoc parsing
  at a 1.00 compliance rate with rescue, and the doctrine there says do not
  build constrained decoding for this.

**Why one call and not qwen-code's two stages**: their stage two is a 3-5 s
cloud CoT, so stage one exists to avoid paying it. Locally the whole
verdict is one short completion; a block that deserves a human gets one via
the fallback, not a second sampling pass.

## The counters (`AgentModeGate`)

Session-scoped (keyed by chat ID), never persisted. `maxConsecutiveBlocks`
= 3: three policy blocks in a row and later calls go straight to the card,
which is what catches an agent retrying variants of a forbidden action.
`maxConsecutiveUnavailable` = 2: the FIRST unavailable already asks (it
does not skip), the second makes later calls skip the known-broken
classifier so an outage stops costing latency.

An allow verdict breaks both streaks. Approving a card breaks both streaks
(qwen-code's recovery rule). Denying a card preserves them. Selecting
Agent mode resets both and lifts a suspension. Chat delete/reset drops the
session's state alongside `SessionApprovalStore`'s.

"Suspend Agent Mode" on a fallback card suspends the classifier for the
rest of the session (the session then asks exactly as `.auto` does).
Re-selecting the mode is the deliberate way out; nothing persists.

## Where the code lives

| File | Role |
|---|---|
| `Tools/Core/AgentMode/ToolCallClassifier.swift` | protocol, verdict, request, projections, `AgentModeHints` |
| `Tools/Core/AgentMode/LocalModelToolClassifier.swift` | prompt, generate + timeout + done-flag, parse |
| `Tools/Core/AgentMode/AgentModeGate.swift` | counters, suspension, thresholds |
| `State/AppModel+AgentMode.swift` | `AgentModeRouting` + the router the loop and batch share |
| `State/AppModel+AgentLoop.swift` | single-call wiring, batch wiring, the park sites |
| `State/SubagentRunner+Gate.swift` | the subagent's classifier path |

The timeout's `GenerationDone` flag is load-bearing: the sleeper only
cancels the session while OUR generate is provably still running, because
a background agent's turn shares that serial queue and a stray
`session.cancel()` at the wrong instant would kill it.

## The subagent shape

A subagent cannot show a card, but it CAN ask the classifier -- the one
scrutiny that needs no UI. Under `.agentAuto`, `permissionRefusal` routes
soft asks through `AgentModeRouting`; a classifier allow runs, a block
refuses with the same message the main loop feeds its model, and an
unavailable verdict refuses exactly as before (fail closed: unattended).
A classifier allow deliberately skips the positive terminal allowlist gate
(`swift/CLAUDE.md` Gotcha 29) on that return: it is a context-aware
verdict, strictly more scrutiny than the allowlist it replaces.

## Settings

`MacAppSettings.agentModeHints` holds the four hint lists (allow /
softDeny / hardDeny / environment), mirrored onto `AppModel.agentModeHints`
like every preference and edited in the Permissions pane
(`AgentModeHintsSectionView`, one sentence per line). Caps apply at PROMPT
time (`AgentModeHints.normalized`: 200 chars per entry, 50 per hint list,
20 environment entries), not at write time, so a hand-edited settings.json
keeps its text on disk. The strings on the mode picker and the card are
English-only by the same convention as the rest of the permission surface
("Approve for me" is not in the catalog either).

## Limits, recorded rather than promised

- The loaded model IS the classifier. A 1B model judges worse than a 27B;
  nothing in the code compensates, same as qwen-code with a weak fast
  model.
- A classification generate pollutes chat-turn KV prefix reuse the same
  way a compaction generate already does; the cost is one extra prefill on
  the next turn. A dedicated small second session is the fix and is
  deliberately NOT built: multi-session is untested in-app.
- A batch of N asks classifies serially (the session queue serializes the
  generates anyway). A single classify-many prompt is the listed
  optimization.
- There is no `classifyAllShell`: a static-allowlisted command never
  reaches the classifier in either direction. Defense-in-depth over the
  fast path would be a new routing arm, not a flag.
