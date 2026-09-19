# Swift session capabilities (reasoning, tool calling, steering)

Facts read off `session.info` after a model opens, not asked for and not
guessed from the family name. This page groups four capability surfaces
that share one shape: a checkpoint-specific truth the ENGINE resolved,
which the UI must read rather than restate.

Read this before touching a reasoning picker, a tool-calling capability
check, or a steering control.

## `SessionInfo` holds what was resolved, not what was asked for

Under automatic sizing nothing was asked for. Read `maxContext` and
`expertCacheSlots` off `session.info` and never off the `OpenOptions` that
produced it; no footprint or throughput figure is readable without the
slot count (root `AGENTS.md` Gotchas 36 and 58). Speculation follows the
same rule: `info.speculation.block != nil` IS the "is it on" test, there is
no second flag that could disagree, and `drafter` is non-nil exactly when
`block` is. Non-nil is a statement about the SESSION, not about the next
turn: acceptance is exact only at temperature 0, so a sampled turn decodes
sequentially whatever it says.

## Reasoning: build the picker from `info.reasoningEfforts`

Not from `Reasoning.allCases` and not from the family. The accepted set
belongs to the checkpoint's own chat template: Qwen 3.8 refuses `.high` and
tops out at `.xhigh` while gpt-oss and Muse Glimmer accept `.high` and have
no `.xhigh`, and a refused level throws from the template mid-turn. The
engine probes that set at open (five renders of a two-line conversation)
and reports it; `reasoningSupport` says only what KIND of control is
meaningful. `AppModel` offered `allCases` for every `.level` checkpoint
until 2026-08-31, so the menu carried an entry that failed the turn.

**Three things that cost something to rediscover.** Levels rendering the
same prompt are collapsed by the engine, so a `.toggleOnly` checkpoint
reports exactly two and its on-level is `.low` BY POSITION -- print "On",
never the spelling (`ReasoningLevelPolicy.label(for:support:)`). There is
no answer at all before a session exists, and the guess that used to fill
the gap was a hardcoded family set, i.e. exactly the table root Gotcha 56
refuses; the controls gate on `AppModel.reasoningPickerEnabled` and the
inspector disables rather than hides, since a missing row reads as a
missing feature. And a preference restored from another model is clamped
to the nearest expressible rung WITH A TOAST, because dropping it to `.off`
silently turns thinking off for someone who turned it on -- the silent
no-op the whole feature exists to avoid.

The decision lives in `ReasoningLevelPolicy`, a pure type: `AppModel.info`
is computed from `session`, so an assertion written against `AppModel`
needs a real install and therefore never runs in the offline suite.

`state#96`: `setReasoning` keyed the remembered level on `alias ?? path`,
so the path arm was dead and two installs sharing an alias shared one
level. Keyed on `path` now, alias read as a legacy fallback.

## Tool calling: read `info.toolCalling.native`, not a family list

This app had three answers to "does this model support tool calls" and
they disagreed; it now has one. `AppModel.isToolCallingSupported` used to
match a hardcoded nine-entry family set and fall through to dialect
substrings; `ModelFeatureDescriptor.supportsToolCalls` was a literal
`true`, driving a "Tool Guardrails" chip that could never read false; and
the fact itself lives in `crates/tokenizer`'s decoder, where three of seven
dialects emit no `ToolCall` at all. Both now read `info.toolCalling.native`,
and `crates/ffi` derives that from ONE `ChatDialect::tool_call_support`
match tied to the decoder by a `debug_assert!` at every site that builds a
call.

**`museGlimmer` was in that hardcoded set and is not native**, which is the
instance worth carrying: it DOES frame tool calls, as an
`<atem:function_calls>` block, and this engine has no parser for it, so the
decoder routes them to the REASONING stream and a caller sees nothing.
Grepping for the markup finds it; grepping for the parser does not. A
family name cannot answer a question about a DIALECT (root `AGENTS.md`
Gotcha 37's shape).

**`native == false` is not a reason to hide a tool control.** It marks the
case a guardrail RESCUE helps most, since recovering a call from raw prose
is the whole first failure `docs/FORGE_GUARDRAILS.md` names. The
guardrails pill gates on whether the turn OFFERS tools
(`interactionMode == .projects` and a non-nil project,
`extractToolCalls`'s own guard), which is the condition under which
`inspect` provably accepts unconditionally.

`isSteeringReady` had the same disease one field over: alias substrings,
missing `gptOss` and `museGlimmer`, which both steer. It now matches the
exact family set from `crates/runtime/src/steering.rs`, with
`info.steering.supported` overriding it whenever a session exists.

`state#97`: `isToolCallingSupported` fell back to `installed.first` with
nothing selected, so an unrelated install decided the Forge guardrails
default; the family list also lagged `crates/model-io`'s by `museGlimmer`,
`qwen4exp` and `deepseekV4Flash`.

## Steering resolves once, at model open

Every `steering*` field in `AppRuntimeOptions` is read by
`buildOpenOptions`, so editing one changes nothing about the model already
loaded -- silently, forever, with the Inspector showing the new value.
`AppSteeringPolicy.needsReload` compares INTENT against `info.steering`, so
it is right in BOTH directions (turning steering off without reloading is
equally a lie), and the Safety pane, the Inspector and the composer pill
all show a reload prompt.

Three more rules the surface is deliberate about. Nothing ships a
direction, so the pane says so and points at `scripts/extract_direction.py`
-- a control labelled as though a behaviour shipped would be a "check the
work exists before adding the control that claims to do it" violation
(`swift/docs/SWIFT_MODEL_HUB.md`'s Probe-button rule). The default strength
is 0.3 and not 1.0, because 1.0 over every layer is a documented collapse
on a real install and a default landing on a documented failure mode is
worse than no default. And the compatibility check says "shape matches",
never "compatible": the engine refuses a width mismatch and a layer
overrun and refuses nothing else, so a vector extracted for another
checkpoint of the same width loads and steers something.
`SteeringPolicyTests` asserts the string does not contain the word
"compatible".

The vector's shape is read through `ts_control_vector_info_json`, not
parsed here. A second GGUF reader in Swift would be free to disagree with
the one the open uses.

## Related: guardrails did not reach the served path

`ForgeGuardrailsEngine` runs in the agent loop over a reply this app read
itself; every HTTP client of the in-process server used to bypass it
entirely and got `ChatModel::guardrails()`'s trait default. So a user who
set "Always Off" and pointed a client at the server got guardrails anyway,
with nothing saying so. `ServerOptions.guardrails` (2026-09-05) carries it,
and `serverStartedGuardrails` records what the START used rather than what
the setting says NOW -- a server resolves its guardrails once and keeps
them, so reporting the live setting would claim a change that did not
happen. A server started before the value was tracked reports `unknown`,
not `on`. `.select` resolves to ON for a server (it means "decide per
project or per chat", and a server request has neither).

This is a different "guardrails" from the memory-loading kind: see root
`AGENTS.md` Gotcha 24 for the naming collision.
