# Tools/Core/AgentMode

## Overview
Agent mode lets a loaded local chat model act as the permission classifier for tool calls, so ordinary calls run without an approval card while risky ones still reach a human. This directory holds the three pieces: the classifier contract and the bounded projection of a call that the classifier sees, the local-model implementation of that contract, and the per-chat counters that decide when to stop consulting the classifier. Design notes live in `swift/docs/SWIFT_AGENT_MODE.md` (referenced from the source comments).

## Interface
- Classifier contract [ToolCallClassifier.swift:ToolCallClassifying, ClassifierVerdict, ClassifierRequest]: one async `classify` that never throws and returns `.allow`, `.block(reason:)`, or `.unavailable(reason:)`.
- Call projection [ToolCallClassifier.swift:ToolCallProjection]: `project(_:)` and `projectedCall(_:)` turn an `AppToolCall` into bounded text per `AppToolCategory`.
- User steering [ToolCallClassifier.swift:AgentModeHints]: Codable allow, softDeny, hardDeny, and environment sentence lists, with `normalized` and `isEmpty`.
- Local implementation [LocalModelToolClassifier.swift:LocalModelToolClassifier]: `init(session:hints:workspaceRoot:)`, `classify`, `parseVerdict(from:)`, and `timeout` (20 s).
- Fallback counters [AgentModeGate.swift:AgentModeGate]: actor with `shared`, `shouldSkipClassifier`, `skipReason`, `isSuspended`, `recordAllow`, `recordBlock`, `recordUnavailable`, `suspend`, `reset`, `clear`, and the thresholds `maxConsecutiveBlocks` (3) and `maxConsecutiveUnavailable` (2).
- Subdirectories: none.

## Invariants
- Classifier failure is never an exception. No model, timeout, or an unparsable reply after one corrective retry all return `.unavailable`, and the caller resolves that as manual approval [LocalModelToolClassifier.swift:classify].
- Fail closed on parsing: a reply is accepted only if it holds a JSON object whose `verdict` is the string "allow" or "block". Anything else yields nil, not an allow [LocalModelToolClassifier.swift:parseVerdict].
- Block reasons are trimmed, replaced with a fixed sentence when empty, and clipped to 400 UTF-8 bytes [LocalModelToolClassifier.swift:parseVerdict].
- Generation is greedy (temperature 0, topK 1, topP 1), reasoning off, capped at 96 new tokens, and must not use constrained decoding (documented as a `docs/SKILL_STATE.md` constraint) [LocalModelToolClassifier.swift:generate].
- On timeout the task group is cancelled but `session.cancel()` is deliberately not called, because the session is shared with the user's chat turn [LocalModelToolClassifier.swift:generate].
- The projection never includes tool results. Shell commands go in full, write content is previewed at 300 bytes, web calls send the URL or query only, per-argument values cap at 2000 bytes under a 16000-byte shared budget, and every cut is marked in place so omission is not read as absence [ToolCallClassifier.swift:ToolCallProjection]. MCP annotations are labeled self-reported and unverified.
- Hint caps (200 bytes per entry, 50 entries per list, 20 environment entries) are applied when building the prompt, not when saving settings, so hand-edited settings keep their text on disk [ToolCallClassifier.swift:AgentModeHints.normalized].
- Gate counters are keyed by chat ID, the same key `SessionApprovalStore` uses, so one chat cannot reset another's counters. An allow resets both counters, `suspend` zeroes them, `reset` also clears suspension, `clear` removes the chat entirely [AgentModeGate.swift:AgentModeGate]. The skip test is `blocks >= 3 || unavailable >= 2`.
- Gate state is in memory only and never persisted; a launch is a fresh session [AgentModeGate.swift].

## Side Effects
- One or two greedy `TurboSparkSession.generate` calls per classification, run through the shared session queue [LocalModelToolClassifier.swift:generate].
- A sleeping child task acts as the 20 s timeout [LocalModelToolClassifier.swift:generate].
- Reads `McpToolCatalogCache.shared` for MCP annotations [ToolCallClassifier.swift:mcpAnnotations].
- `AgentModeGate` mutates actor-isolated in-memory dictionaries and a set [AgentModeGate.swift].
- No disk, network, Keychain, UserDefaults, or process spawn in this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
