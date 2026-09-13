# Native agent efficiency

TurboSparkApp implements four local efficiency mechanisms. They are native
Swift behavior, not a Pi dependency, and all start enabled for existing and
new chats. Each has an individual switch in Settings, General, Agent
Efficiency.

## Fused mutation validation

`FileWrite`, `FileEdit`, and `apply_patch` accept optional flat arguments:

```text
validate_command: string
validate_timeout_ms: integer
```

The agent loop derives a `run_command` validation leg before a mutation. Both
legs pass through PreToolUse hooks, risk classification, and permission
evaluation independently before either can run. A mutation hook's allow never
authorizes an unresolved validation hook. One approval card combines the
mutation and terminal risks. A preflight denial blocks both legs. After a
successful mutation, the declared file targets are fingerprinted before the
command runs.
Validation failure is recorded but does not undo a successful mutation.

`apply_patch` fingerprints every declared add, update, and delete target. The
transcript keeps the original mutation call and adds a compact fused validation
outcome to its card. See [SWIFT_TOOLS.md](SWIFT_TOOLS.md) for the broader tool
contract.

## Tool observations and recall

Large normal-chat tool output is kept exactly under the active profile's
`tool-observations` store until the owning chat is deleted. The message result
continues to carry the ordinary UI and export output. Its model-facing
projection is separate.

After two real prompt sends of a full archived result, the projection becomes a
stable head-and-tail placeholder. The model can use:

```text
recall_tool_output(observation_id, offset_bytes, max_bytes)
```

to retrieve bounded, exact byte ranges. Prompt assembly and the context meter
consume this same projection. A full result that cannot fit the active context
uses the placeholder immediately.

Ghost chat observations never enter the profile store. They remain inside the
encrypted, in-memory Ghost vault and disappear with the chat or app process.

## Verified local reduction

Eligible long terminal diagnostics can be reduced only by the active local
session. The reducer receives deterministic settings and must return JSON with
the source SHA-256, terminal status, a bounded summary, and bounded exact
quotes. TurboSpark verifies the hash, status, quote lengths, and byte-source
membership before accepting a receipt. Any malformed response, cancellation,
or local model error leaves the ordinary observation projection in place.

No remote model, provider, route, or credential is introduced for reduction.

## Completed-todo boundaries

When `TodoWrite` changes an item to completed while unfinished work remains,
TurboSpark can reuse the existing compaction summary and continuation path
before the next agent step. It compacts only when there is a net history
benefit. The normal pressure trigger at 80 percent of usable context remains
unconditional and unchanged. See [SWIFT_COMPACTION.md](SWIFT_COMPACTION.md).
