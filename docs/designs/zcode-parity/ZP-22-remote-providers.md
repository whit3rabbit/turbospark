# ZP-22: Remote model providers (upstream APIs)

Findings 7.1 and 7.2 (API-key half) in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
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
Adopted by author decision 2026-11-11. Cloud OAuth is excluded
permanently by the same decision; API keys are the only credential type.

## The Problem

Chat always runs on local weights. When a task exceeds what the machine
can hold or the user wants a frontier model for one reply, there is no
path; the app's own server code proves the codebase already speaks both
API families, but only in the serving direction. ZCode treats remote
providers as its primary model source with presets, custom endpoints,
per-provider keys, and editable model lists.

## Goals

* A Providers settings section: named providers, each with an API family
  (anthropic, openai-compatible), base URL, API key, and an editable
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
* Secrets follow the repo rules: per-credential opt-in Keychain items,
  never at launch, never in settings.json, never logged.

## Non-Goals

* **No OAuth, ever.** No account login, no hosted identity, no provider
  family switching. Author decision 2026-11-11: API keys only.
* No hosted TurboSpark service, no billing, no quota-reset flows
  (findings 7.3/7.4 stay excluded; per-provider usage endpoints are a
  possible follow-up, not a goal here).
* No remote model downloads or weight handling; providers serve tokens,
  not checkpoints.
* No automatic failover from local to remote; selection is always
  explicit.

## Open Questions

* **Tool loop phase order**: chat-only first with the tool loop in a
  fast-follow, or both in one landing? The Swift turn loop already
  parses native tool calls for the local engine; if that loop is
  provider-agnostic at the "model step" seam, remote tools are a thin
  adapter and land together. Verify the seam first; if the loop is
  FFI-coupled, land chat-only and split tools out. This verification is
  the first implementation task.
* **Context accounting for remote chats**: exact local token counting is
  impossible for provider tokenizers. Options: (a) API usage watermarks
  only (ring shows last reported usage; pre-send estimate uses a
  chars-per-token heuristic); (b) ship per-family approximate tokenizers.
  Lean (a) with the estimate labeled as an estimate in the ring popover.
* **Connection checks**: a "Test connection" button per provider (one
  cheap models-list or tiny-completion call) mirroring ZCode's
  connectivity results. Include; settle exact probe call at
  implementation.

## Critical User Journeys

### CUJ 1: configure a provider

* **Given**: the user opens Settings, Providers.
* **When**: they add a provider (name, family, base URL), paste an API
  key, and add a model entry.
* **Then**: the key is stored as a Keychain item (opt-in already
  satisfied by the act of entering it; a "store in Keychain" toggle
  defaults on and off means the provider stays disabled), the model
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
* **Then**: the call passes the ordinary permission ladder and executes
  exactly as a local-model tool call; results return in the family's
  tool-result format; nothing about being remote changes approvals.

### CUJ 4: secret hygiene

* **Given**: a configured provider.
* **When**: the user exports settings or inspects logs.
* **Then**: settings.json holds provider shape (name, family, URL,
  models) but never the key; logs redact Authorization headers and any
  key-shaped strings; quitting and relaunching does not read the key
  until a remote model is actually used.

## Technical Plan (Architecture)

```text
Settings > Providers
  ProviderStore (new): [Provider], [ProviderModel]
  key -> Keychain (existing credential rules), never settings.json
       |
       v
Composer model picker
  Local section (existing) | Remote section (ProviderStore)
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

Compaction works unchanged: the summarizing call runs on the chat's
selected model, local or remote. Window fitting and expert residency are
local-only concerns and stay untouched.

## Alternatives Considered

### Alternative A: OAuth account-first like ZCode

* **Pros**: frictionless for Z.ai-style plans.
* **Cons**: permanently excluded by author decision; hosted identity
  contradicts the local-first posture.
* **Reason for Rejection**: Rejected by author: never cloud OAuth.

### Alternative B: route remote calls through the local Rust server

* **Pros**: one HTTP stack; FFI symmetry.
* **Cons**: a loopback hop for a client-side concern; the turn loop and
  steering already live Swift-side; SSE clients in Swift are small
  against the FFI/wire/binding cost.
* **Reason for Rejection**: Rejected by author: the Swift-side client
  keeps the turn loop unified without new FFI surface. Revisit only if a
  Rust-side need (for example CLI parity) appears.

### Alternative C: proxy-style "remote fallback" that silently picks local or remote

* **Pros**: hides configuration.
* **Cons**: silent model selection breaks trust and reproducibility;
  quality and privacy differ between paths.
* **Reason for Rejection**: Rejected by author: explicit selection only.

## Non-Functional Requirements

* **Security and privacy**: prompts sent to a remote provider leave the
  machine; the provider settings surface and the first remote send in a
  chat both state this plainly. Keys: Keychain only, never logged, never
  in exports; a log redaction pass covers key-shaped strings. Remote
  chats are excluded from any future cloud surface by the same rules.
* **Performance**: streaming parses off the main thread; one in-flight
  request per chat; request timeout and stall detection with named
  errors; no polling.
* **Reliability**: named error taxonomy per family (auth, rate limit,
  overloaded, network, bad stream) surfaced in the transcript row and
  the toast, matching local engine error handling.
* **Localization**: all settings and error strings in the catalog, 21
  languages.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/ProviderStore.swift` (Create)
   * **Type / Scope**: provider and model configuration.
   * **Proposed Changes**: provider/model types, JSON persistence via
     AppJSONStore (shape only, no keys), Keychain key items per the
     existing credential rules, enable state.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Networking/AnthropicClient.swift` (Create)
   * **Type / Scope**: family client.
   * **Proposed Changes**: /v1/messages SSE streaming, tool schemas,
     usage and finish parsing, named error taxonomy, interrupt support.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Networking/OpenAICompatClient.swift` (Create)
   * **Type / Scope**: family client.
   * **Proposed Changes**: /chat/completions SSE streaming with
     tool_calls deltas, same event emission contract.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ModelStep.swift` (Create; the seam type) plus the turn-loop integration in the existing generation state files (Modify; basenames: the AppModel generation extension files)
   * **Type / Scope**: turn loop seam.
   * **Proposed Changes**: extract the per-step model call behind a
     protocol; LocalModelStep wraps the existing FFI path unchanged;
     RemoteModelStep dispatches to the family client; downstream
     (steering, queues, tool ladder, compaction) untouched.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/App/SettingsPanes.swift` (Modify; basename, pane container) plus a new `ProviderSettingsPaneView.swift` (Create)
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
     remote steps.
   * Mutation checks: log a key in a debug path and assert the redaction
     test fails; let a remote tool call skip the ladder and assert the
     ladder-equality test fails.
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
* **Localization**: catalog additions in 21 languages; ten parity gates.
* **Manual**: chat against a local mock SSE server (both families) for
  deterministic checks; one real-provider smoke by the user; verify key
  absence from settings.json and logs; verify interrupt, steering, and
  tool approval on a remote chat.

--------------------------------------------------------------------------------

*Drafted with EGM*
