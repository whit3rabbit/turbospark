# Guardrails

## Overview
`ForgeGuardrailsEngine` is the tool-call repair layer between raw model output and the agent loop. Given the generated text, the calls the standard parser found, and the tools offered this turn, it returns one of three verdicts: accept the turn, rescue tool calls out of raw text markup the standard parser missed, or retry with a corrective nudge. The directory holds a single file with the verdict type, the dialect rescue patterns, argument validation, and prose sanitizing.

## Interface
- `ForgeGuardrailVerdict`: `.accept`, `.rescued(calls:sanitizedText:)`, `.retry(nudge:)` [ForgeGuardrailsEngine.swift:ForgeGuardrailVerdict].
- `ForgeGuardrailsEngine.inspect(text:parsedCalls:availableTools:requiresCall:projectURL:)`: the entry point [ForgeGuardrailsEngine.swift:inspect].
- `ForgeGuardrailsEngine.rescueToolCalls(from:availableToolNames:projectURL:)`: dialect rescue (Qwen/XML function and invoke tags, GLM `arg_key`/`arg_value` pairs, Kimi K2 section tags, Gemma `call:NAME{...}`, Mistral `[TOOL_CALLS]`, Markdown code fences, Longcat tags, bare JSON) [ForgeGuardrailsEngine.swift:rescueToolCalls].
- `ForgeGuardrailsEngine.validateArguments(arguments:schema:)`: required, integer, number, boolean, and enum checks [ForgeGuardrailsEngine.swift:validateArguments].
- `ForgeGuardrailsEngine.sanitizeProse(text:rescuedCalls:)`: strips rescued markup and wrapper markers [ForgeGuardrailsEngine.swift:sanitizeProse].
- Internal helper `isToolNameAllowed(_:in:)`.
- Child directories: none.

## Invariants
- Every rescue path must gate on `isToolNameAllowed` (case-insensitive membership in the tools actually granted this turn) before creating an `AppToolCall`. A rescued call is indistinguishable from a model-issued one, so skipping the check would let text grant a tool the agent was never given. An earlier `|| !name.isEmpty` bug made the check pass for any name [ForgeGuardrailsEngine.swift:isToolNameAllowed, rescueToolCalls].
- Rescue is tried only when standard parsing found no calls. Dialect patterns are tried in order and the first pattern that yields calls wins (XML, GLM, Kimi, Gemma, Mistral, Markdown, Longcat, then bare JSON). Reordering changes which dialect claims ambiguous text [ForgeGuardrailsEngine.swift:rescueToolCalls].
- Matches preceded by an incomplete tool-call label are skipped via `ToolCallParser.isPrefixedByIncompleteToolCallLabel`, so a partial streamed label is not rescued early [ForgeGuardrailsEngine.swift].
- Rescued calls start as `.pendingApproval` with category from `AppToolRegistry.category(for:projectURL:)` and risk from `ToolRiskClassifier.assessRisk(..., projectURL:)`. `projectURL` must be passed so a project-scoped custom tool is not rated as low-risk automation. Rescue never bypasses approval [ForgeGuardrailsEngine.swift].
- Tool specs are indexed with `uniquingKeysWith` (first wins). Do not change it to `uniqueKeysWithValues`: a project custom tool may shadow a built-in name and the trap would crash the app [ForgeGuardrailsEngine.swift:inspect].
- An empty tool list always yields `.accept`. With no calls and `requiresCall` true, the verdict is `.retry` with a nudge naming the sorted tool list. Any unknown tool name or schema violation yields `.retry` with all errors joined; validation runs before the rescued verdict is returned [ForgeGuardrailsEngine.swift:inspect].
- `validateArguments` accepts booleans only as `true` or `false`, and an empty-string required value counts as missing. `flattenJsonToDict` emits `true`/`false` for JSON booleans (not 1/0) for this reason. Nested JSON values are re-serialized to JSON strings [ForgeGuardrailsEngine.swift:validateArguments, flattenJsonToDict].
- `sanitizeProse` returns an empty string when only bare punctuation or an empty fence remains [ForgeGuardrailsEngine.swift:sanitizeProse].
- The Rust server needs no Longcat-specific code (its JSON scan reads balanced braces anywhere), but this Swift Pattern 4 requires the whole text to parse as JSON, so the Longcat wrapper must be stripped first [ForgeGuardrailsEngine.swift].

## Side Effects
- None. The engine is a set of pure static functions over text, parsed calls, and tool schemas, with no disk, network, Keychain, UserDefaults, process spawn, NotificationCenter, timers, or tasks. It only calls into `AppToolRegistry.category` and `ToolRiskClassifier.assessRisk` for classification [ForgeGuardrailsEngine.swift].

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
