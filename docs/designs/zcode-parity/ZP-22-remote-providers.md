# ZP-22: Remote model providers (upstream APIs)

Findings 7.1 and 7.2 (API-key half), with 7.14 (credential rules), 7.15
(composer picker), and 7.16 (connectivity checks) in
[ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[ModelProviderSection](https://github.com/zai-org/ZCode/blob/main/packages/ui/src/settings/ModelProviderSection.tsx)
and the
[provider JSON schema doc](https://github.com/zai-org/ZCode/blob/main/docs/model-provider-json-schema.md).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Add opt-in remote providers: the app can chat through upstream APIs
(Anthropic and OpenAI-compatible endpoints) with user-managed API keys,
alongside local weights. Local stays the default and the product center;
remote models appear as a clearly labeled section in the composer picker.
Adopted and scoped by the 2026-11-11 author decision recorded in
[ZP-90](ZP-90-excluded-and-decision-findings.md) (7.1 adopted, 7.2
excluded). API keys are the only credential type.

## Problem Statement

Chat always runs on local weights. When a task exceeds what the machine
can hold or the user wants a frontier model for one reply, there is no
path; the app's server already serves both endpoint shapes
(ZCODE_COMPARISON 7.13), but only in the serving direction. ZCode treats remote
providers as its primary model source with presets, custom endpoints,
per-provider keys, and editable model lists.

## Goals

* A Providers settings section: named providers, each with an API family
  (anthropic, openai-compatible; an API family is the wire protocol,
  persisted as `anthropic` or `openai-compatible`, and the class names
  `AnthropicClient` and `OpenAICompatClient` are the family clients; prose
  uses "OpenAI-compatible"), base URL, API key, and an editable
  model list (id, display name, context window, vision and reasoning
  capability flags, enabled toggle).
* The composer model picker gains a Remote section listing enabled
  provider models, visually distinct from local models; local remains
  the default selection everywhere.
* A streaming chat client per family (SSE) that drives the existing
  Swift turn loop: content, tool calls, usage, finish reason, and
  interrupt semantics behave as they do for local chats.
* Agent tool calling works against remote models through the existing
  tool registry and permission ladder; no tool bypasses the ladder
  because the model is remote.
* Usage accounting records remote turns in the existing per-chat usage
  ledger, provider-tagged; context usage for remote chats uses API
  usage watermarks plus a documented local estimate for the pre-send
  ring.
* A Test connection button per provider reports success or a named
  failure (auth, network, endpoint).
* Secrets follow the repo rules (root AGENTS.md and ZCODE_COMPARISON
  7.14): per-credential opt-in Keychain items, never at launch, never in
  settings.json, never logged. Storage follows the `ServerKeychain`
  pattern (`Server/ServerKeychain.swift`, `Server/KeychainRead.swift`);
  see ZCODE_COMPARISON 7.14.

## Non-Goals

* **No OAuth.** No account login and no hosted identity; API keys are the
  only credential type (decision in ZP-90). A provider's family is fixed
  when it is created; changing family means adding a new provider.
  [confirm that is the intended meaning of "family switching"]
* No hosted TurboSpark service, no billing, no quota-reset flows
  (findings 7.3/7.4 (hosted billing and quota reset, excluded in ZP-90);
  per-provider usage endpoints are a possible follow-up, not a goal
  here).
* No provider presets, no import of ZCode's provider JSON or another
  tool's provider config, and no settings sync; providers are entered by
  hand in this change. A later spec may add import. [confirm]
* No remote model downloads or weight handling; providers serve tokens,
  not checkpoints.
* No automatic failover from local to remote; selection is always
  explicit.
* No background feature (titles, memory, cron, microcompact) selects a
  remote model on its own; each uses the model the chat selected, and a
  feature with no chat context stays local.
* The TurboSpark API server never proxies to a remote provider, and
  provider keys are never visible to it.
* Known limit: "OpenAI-compatible" endpoints differ in vision input,
  reasoning fields, and usage reporting. The client implements the common
  `/chat/completions` subset; a field a provider omits (for example usage)
  degrades to the estimate and is labeled. No per-vendor quirk profiles.

**Dependencies**: the rollout-log entry in CUJ 2 needs ZP-03; without
ZP-03 remote turns skip that step (soft dependency). ZP-22 does not block
ZP-23 (concurrent chats); one in-flight request per chat holds either way.
[confirm ZP-23 interaction] Separately, `INDEX.md` line 89 should read
"has a soft dependency on ZP-03 and touches" instead of "stands alone but
touches" (outside this file; the author applies it).

## Critical User Journeys

### CUJ 1: configure a provider

* **Given**: the user opens Settings, Providers.
* **When**: they add a provider (name, family, base URL), paste an API
  key, and add a model entry.
* **Then**: the key is stored as a generic-password Keychain item when
  the user saves it; saving is the opt-in for that credential, and there
  is no separate toggle. Until saved, the provider is disabled. The key is
  read from the Keychain only at the first send to that provider in a
  session, never at launch. A denied or failed read sends nothing and
  shows a named error (for example "Keychain access denied"). The model
  appears in the composer's Remote section, and Test connection reports
  success or a named failure.

### CUJ 2: chat with a remote model

* **Given**: a remote model is selected in the composer.
* **When**: the user sends a prompt.
* **Then**: the reply streams in with the same transcript rendering,
  steering, queueing, and stop semantics as local chats; the usage
  ledger records the turn provider-tagged; the context ring shows the
  API-reported usage; the rollout log (ZP-03) records the call with a
  remote marker.

### CUJ 3: tool call against a remote model

* **Given**: the agent loop runs with a remote model.
* **When**: the model emits a tool call (family-native tool schema).
* **Then**: the call passes the ordinary permission ladder (the app's
  tiered tool approval, deny or ask by default; see
  `swift/docs/SWIFT_TOOLS.md` [confirm anchor]) and executes
  exactly as a local-model tool call; results return in the family's
  tool-result format; nothing about being remote changes approvals.

### CUJ 4: secret hygiene

* **Given**: a configured provider.
* **When**: the user inspects settings.json, a profile export or backup,
  or logs.
* **Then**: none of them contains the key; settings.json holds provider
  shape (name, family, URL, models); logs redact Authorization headers and
  any key-shaped strings; quitting and relaunching does not read the key
  until a remote model is actually used. [confirm ATS: https-only needs no
  exception; loopback http needs an ATS local-networking allowance, check
  `Info.plist`]

## Open Questions

* **Connection probe call**: models-list (free, but not served by every
  compatible endpoint) versus a one-token completion (costs tokens,
  universal). Lean models-list with a completion fallback. [confirm]

## Proposed Architecture

```text
Settings > Providers
  RemoteProviderStore (new): [Provider], [ProviderModel]
  key -> Keychain (existing credential rules), never settings.json
       |
       v
Composer model picker
  Local section (existing) | Remote section (RemoteProviderStore)
       |
       v
Turn loop seam (existing Swift orchestration)
  LocalModelStep (existing FFI path)
  RemoteModelStep (new): family client
    AnthropicClient   /v1/messages SSE
    OpenAICompatClient /chat/completions SSE
  both emit the same step events the loop already consumes
  (content deltas, tool calls, usage, finish, interrupt)
       |
       v
Shared downstream: permission ladder, steering, queues,
usage ledger, compaction, rollout log (provider-tagged)
```

**Turn loop seam (verified 2026-11-11)**: the loop in
`State/AppModel+Generation.swift` calls the engine through
`AppStreamRecovery.generateUntilComplete` and carries engine-typed results
(`GenerationResult`, `GenerationToolCall`). It is not provider-agnostic
today. This design introduces `ModelStep` as a protocol returning
engine-neutral step events (content delta, tool call, usage, finish,
interrupt); `LocalModelStep` adapts the existing path and maps
`GenerationToolCall` into the neutral type. Downstream behavior (steering,
queues, ladder, compaction) is unchanged in behavior, but the call sites
listed in `AppModel+Generation.swift` and `AppModel+AgentLoop.swift`
change type. [Author to confirm the exact call-site list after a read of
both files.]

**Context accounting and compaction**: decision, option (a). The ring
shows the last API-reported usage; the pre-send estimate is characters
divided by a fixed constant (4, labeled "estimate" in the ring popover)
[confirm constant]. Compaction for remote chats triggers on the
API-reported watermark or the estimate against the model entry's context
window, not on `session.countTokens`. A provider context-overflow error is
a named error that offers compaction. The summarizing call runs on the
chat's selected model. [confirm trigger threshold reuse from
SWIFT_COMPACTION.md] Window fitting and expert residency are local-only
concerns and stay untouched.

## Alternatives Considered

### Alternative A: OAuth account-first like ZCode

* **Pros**: frictionless for Z.ai-style plans.
* **Cons**: permanently excluded by author decision; hosted identity
  contradicts the local-first posture.
* **Reason for Rejection**: Rejected: hosted identity contradicts
  local-first and ZP-90 excludes cloud OAuth permanently.

### Alternative B: route remote calls through the local Rust server

* **Pros**: one HTTP stack; FFI symmetry.
* **Cons**: a loopback hop for a client-side concern; the turn loop and
  steering already live Swift-side; SSE clients in Swift are small
  against the FFI/wire/binding cost.
* **Reason for Rejection**: Rejected: adds a loopback hop and new FFI
  surface for a client-side concern; revisit if CLI parity needs it.

### Alternative C: proxy-style "remote fallback" that silently picks local or remote

* **Pros**: hides configuration.
* **Cons**: silent model selection breaks trust and reproducibility;
  quality and privacy differ between paths.
* **Reason for Rejection**: Rejected: silent selection breaks trust and
  reproducibility; see Non-Goals.

## Non-Functional Requirements

* **Security and privacy**: prompts sent to a remote provider leave the
  machine; the provider settings surface states this plainly. The first
  remote send per provider per chat shows a notice naming the provider
  host and stating "prompts and tool results leave this machine"; the
  send waits for confirmation. Switching a chat from local to remote
  mid-chat triggers the same notice. [confirm notice wording and whether
  confirmation blocks] Keys: Keychain only, never logged, never
  in exports; a log redaction pass covers key-shaped strings. Base URLs
  must be https; plain http is accepted only for loopback hosts
  (localhost, 127.0.0.1, [::1]). Any other http URL is refused at save
  time. Redirects to a different host drop the Authorization and
  x-api-key headers (reuse `HttpRequestRedirectDelegate`) or are refused.
  Redaction patterns: header values for Authorization and x-api-key, and
  strings matching `sk-` and `sk-ant-` prefixes followed by 16 or more key
  characters. [confirm patterns and the minimum length]
* **Performance**: streaming parses off the main thread; one in-flight
  request per chat; request timeout and stall detection with named
  errors; no polling.
* **Reliability**: named error taxonomy per family (auth, rate limit,
  overloaded, network, bad stream) surfaced in the transcript row and
  the toast, matching local engine error handling. One first-byte timeout
  and one stall timeout (values fixed at implementation, stated in
  `SWIFT_REMOTE_PROVIDERS.md`). No automatic retry: every error ends the
  turn with its named error and the user resends. Each named error (auth,
  rate limit, overloaded, network, bad stream, context overflow) maps onto
  the local engine's error row type. [author to confirm the mapping target
  after reading the local error row code]
* **Localization**: all settings and error strings in the catalog, in
  [21 languages and the parity gates](../../../swift/docs/SWIFT_LOCALIZATION.md).

## Risks and Mitigations

* **Provider API drift** (field or event changes): pin tested request
  shapes in SSE fixtures; unknown event types are ignored, not fatal.
  [confirm]
* **Tool-result content leaves the machine**: tool output returned to a
  remote model is sent to the provider; the first-send notice says so, and
  tool approvals are unchanged.
* **Stream truncated mid tool call**: partial tool JSON is discarded, the
  turn ends with the named "bad stream" error, no tool runs.
* **Base URL misconfiguration or leakage**: https required (loopback
  excepted), Authorization stripped on cross-host redirect (see Security
  and privacy above).
* **Remote loop cost**: a runaway agent loop costs money; the usage ledger
  shows per-provider totals, and the existing step cap applies. [confirm
  that a step cap exists for remote chats]

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/RemoteProviderStore.swift` (Create)
   * **Type / Scope**: provider and model configuration.
   * **Proposed Changes**: provider/model types, JSON persistence via
     AppJSONStore (shape only, no keys), Keychain key items per the
     existing credential rules, enable state; owns the Keychain read and
     write for provider keys and the redaction helper that the clients
     call before any log write; the redaction helper is the single egress
     point for provider logging.
2a. `swift/TurboSparkApp/Sources/TurboSparkApp/Networking/SSEParser.swift` (Create; `Networking/` is a new directory)
   * **Type / Scope**: shared transport parsing.
   * **Proposed Changes**: shared line-oriented SSE parser used by both
     clients.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Networking/AnthropicClient.swift` (Create)
   * **Type / Scope**: family client over the shared SSEParser; transport
     reuses `HttpRequestRedirectDelegate` from
     `Tools/Web/HttpRequestExecutor.swift`; the private-address refusal is
     not applied to user-configured provider hosts [confirm].
   * **Proposed Changes**: /v1/messages SSE streaming, tool schemas,
     usage and finish parsing, named error taxonomy, interrupt support.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Networking/OpenAICompatClient.swift` (Create)
   * **Type / Scope**: family client over the shared SSEParser; same
     transport reuse and [confirm] as item 2.
   * **Proposed Changes**: /chat/completions SSE streaming with
     tool_calls deltas, same event emission contract.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ModelStep.swift` (Create; the seam type) plus `State/AppModel+Generation.swift` and `State/AppModel+AgentLoop.swift` (Modify). Consider placing `ModelStep.swift` under `State/` instead of `Generation/` (which holds views); [confirm].
   * **Type / Scope**: turn loop seam.
   * **Proposed Changes**: extract the per-step model call behind a
     protocol; LocalModelStep wraps the existing FFI path unchanged;
     RemoteModelStep dispatches to the family client; downstream
     (steering, queues, tool ladder, compaction) unchanged in behavior.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/App/SettingsTab.swift` and `App/AppSettingsView.swift` (Modify: add the Providers tab) plus `Components/RemoteProviderSettingsPaneView.swift` (Create)
   * **Type / Scope**: settings UI.
   * **Proposed Changes**: provider list editor, key entry (Keychain),
     model list editor, Test connection, privacy notice text.
6. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/PromptContextPills.swift` (Modify)
   * **Type / Scope**: picker.
   * **Proposed Changes**: Remote section with provider grouping and
     distinct styling; local default preserved.
7. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppChatUsage.swift` (Modify)
   * **Type / Scope**: usage ledger.
   * **Proposed Changes**: provider-tagged entries for remote turns.
8. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppContextUsage.swift` (Modify)
   * **Type / Scope**: context ring.
   * **Proposed Changes**: API watermark path plus labeled estimate for
     remote chats.
9. `swift/TurboSparkApp/Tests/TurboSparkAppTests/RemoteProviderTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: SSE fixture parsing per family (content,
     tool calls, usage, errors, mid-stream interrupts); key never
     persists to settings.json; log redaction; usage ledger tagging;
     picker gating (Remote section hidden with no providers); unconfigured
     and disabled error paths; tool ladder equality between local and
     remote steps; 401 and 429 responses, disconnect mid-stream, truncated
     tool-call JSON.
10. `swift/docs/SWIFT_REMOTE_PROVIDERS.md` (Create)
    * **Type / Scope**: subsystem doc.
    * **Proposed Changes**: architecture, family support matrix, secret
      rules, error taxonomy, and the no-OAuth decision record.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for RemoteProviderTests plus existing
  generation, usage, and settings tests.
* **Static Analysis and Linting**: `make design-check` for the new pane
  and picker styling.
* **Localization**: catalog additions in
  [21 languages and the parity gates](../../../swift/docs/SWIFT_LOCALIZATION.md).
* **Mutation checks**: log a key in a debug path and assert the redaction
  test fails; let a remote tool call skip the ladder and assert the
  ladder-equality test fails. Assert the mutation applied before treating
  a survivor as evidence.
* **Manual**: chat against a local mock SSE server (both families) for
  deterministic checks; one real-provider smoke by the user; verify key
  absence from settings.json and logs; verify interrupt, steering, and
  tool approval on a remote chat.

--------------------------------------------------------------------------------

*Drafted with EGM*
