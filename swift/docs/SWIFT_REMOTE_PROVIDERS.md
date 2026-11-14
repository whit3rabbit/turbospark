# Remote providers (ZP-22)

Opt-in chat through upstream APIs with user-managed API keys. Local weights
stay the default and the product center. Design and review record:
`docs/designs/zcode-parity/ZP-22-remote-providers.md`.

**Status: implemented.** Remote stays off by default: with no provider
configured nothing here runs, the composer shows no Remote control, and every
chat uses the local model. The one manual follow-up is a real-provider smoke
test by the user (the tests use scripted transports only).

## Decision record: API keys only, no OAuth

Adopted 2026-11-11 (ZP-90). Cloud OAuth and hosted identity are excluded
permanently: they contradict the local-first posture. A provider's API family
is fixed at creation. Changing family means adding a new provider.

## Files

| File | Role |
| --- | --- |
| `Networking/SSEParser.swift` | Byte-level `text/event-stream` parser (CRLF, comments, multi-line data, split UTF-8, per-event cap). |
| `Networking/RemoteProviderTypes.swift` | `RemoteAPIFamily`, `RemoteProvider`, `RemoteProviderModel`, `RemoteProviderError`, step events, request, `RemoteLimits`. |
| `Networking/RemoteEndpointPolicy.swift` | Base URL rules, endpoint joining, redirect rule. |
| `Networking/RemoteProviderRedactor.swift` | The single egress point for provider text (exact key plus `TrajectoryRedactor` patterns). |
| `Networking/RemoteHTTPTransport.swift` | The one network seam and the URLSession transport. |
| `Networking/RemoteStreamRunner.swift` | Shared redirect, size, time, error and probe logic. |
| `Networking/AnthropicClient.swift`, `OpenAICompatClient.swift` | Family request builders and stream decoders. |
| `State/RemoteProviderKeyStore.swift` | Per-credential key handling and the Keychain backend. |
| `State/RemoteProviderStore.swift` | Provider and model shape (no keys), picker entries, per-chat consent gate. |
| `State/RemoteProviderService.swift` | The one entry point: gates, key read, runner. |
| `State/AppModel+RemoteTurn.swift` | Selection, the remote turn (`executeRemoteGenerationTurn`), mapping to `GenerationResult`, remote context and compaction, rollout call, localized error text. |
| `Components/RemoteProviderSettingsPaneView.swift` | Settings > Network > Remote providers. |
| `Generation/RemoteModelPill.swift` | Composer Remote menu, persistent indicator, first-send confirmation. |

## Family support matrix

| | anthropic | openai-compatible |
| --- | --- | --- |
| Endpoint | `/v1/messages` (a base ending in `/v1` is not doubled) | `/chat/completions` |
| Auth header | `x-api-key` plus `anthropic-version` | `Authorization: Bearer` |
| Stream | named SSE events | `data:` chunks, `[DONE]` |
| Tool calls | `tool_use` blocks, `input_json_delta` | `tool_calls` deltas by index |
| Usage | `message_start` plus `message_delta` | final chunk (`stream_options.include_usage`) |
| Reasoning | `thinking_delta` | `reasoning_content` or `reasoning` |
| Probe | `GET /v1/models` | `GET /models` |

OpenAI-compatible endpoints differ in vision input, reasoning fields and usage
reporting. The client implements the common subset only. A field a provider
omits (usage) degrades to the labeled estimate. There are no per-vendor
profiles.

## Secret rules

* Keys are never stored by default. A key typed in Settings lives in memory
  for the process.
* Keychain storage is opt-in per credential: the explicit "Save to Keychain"
  action writes a generic-password item (service `TurboSpark.remote-provider`,
  account per provider id) and records the opt-in in `UserDefaults` for that
  provider only. Forgetting deletes both.
* The Keychain is never touched at launch, in `init`, or while building the
  picker (`isKeyAvailable` is an attributes-only probe). The secret is read at
  the first send to that provider in a session and then held in memory.
* A failed read is `RemoteProviderError.keyUnavailable` (named, with the
  system's reason). It sends nothing and never deletes the stored item.
* No key in `remote_providers.json` (`RemoteProvider` has no key field), in
  `settings.json`, in a URL (a query string in a base URL is refused), in a
  log, or in error text. Everything provider-derived passes
  `RemoteProviderRedactor`: the exact key, `Authorization` and `x-api-key`
  header values, and `sk-` and `sk-ant-` shaped tokens of 16 or more key
  characters.

## Egress rules

* Base URLs must be `https`. `http` is accepted for loopback only
  (`localhost`, `127.0.0.1`, `[::1]`, `*.localhost`). No userinfo, query or
  fragment. Checked at save and again before every send.
* The private-address refusal used for agent web fetches is deliberately not
  applied: the user configured the host (a LAN gateway is legitimate). What
  replaces it is explicitness (below).
* Redirects: the transport never follows one. The runner follows a hop only to
  the same host and port, never https to http, at most 3 hops, and rebuilds
  the request (so the key header is attached per hop by us). Anything else is
  `redirectRefused` and the target is never contacted. The capture uses the
  shared `HttpRequestRedirectDelegate`.
* Caps (`RemoteLimits.standard`): request body 8 MiB (refused before send),
  response stream 32 MiB, one SSE event 1 MiB, error body 16 KiB, first-byte
  timeout 60 s, stall timeout 90 s. No automatic retry.
* The session is ephemeral: no cookie jar, no URL cache.
* Explicit per chat: a provider is usable only with a key and an enabled
  model, and the first send per provider per chat waits for a confirmation
  that names the host and says prompts and tool results leave this machine
  (`RemoteConsentLedger`, in memory, so a relaunch asks again). Remote use is
  off by default: no provider means no Remote section.
* The TurboSpark API server never proxies to a remote provider and never sees a
  provider key.

## Error taxonomy

`RemoteProviderError`: `notConfigured`, `disabled`, `invalidBaseURL`,
`keyUnavailable`, `auth` (401, 403), `endpointNotFound` (404), `rateLimited`
(429), `overloaded` (503, 529), `contextOverflow` (413, or 400 and 422 naming
a context limit, offers compaction), `server` (other 5xx), `requestRefused`
(other 4xx, redacted and clipped detail), `network`, `badStream`,
`redirectRefused`, `firstByteTimeout`, `stalled`, `responseTooLarge`,
`requestTooLarge`, `cancelled`. A stream cut mid tool call is `badStream` and
no call is emitted: tool arguments are validated as one JSON object at block
end, and OpenAI-style calls only at the end of the stream.

The case-to-localized-string and case-to-transcript-row mapping belongs to the
UI increment. `diagnostic` is English text for logs and tests.

## Verification

`RemoteProviderTests.swift` uses a scripted transport and a fake Keychain
backend. No test opens a socket or touches the real Keychain. It covers SSE
fixtures per family (content, reasoning, tool calls, usage, errors, truncated
tool JSON, missing terminal event), status mapping, key redaction, cross-host
and downgrade redirects (target never contacted), caps, stall and first-byte
timeouts, mid-stream disconnect, probe fallback, Keychain opt-in (no touch at
init or picker build, one read at first send, named failure, no delete on a
failed read), shape file without keys, and the consent gate. Mutation checks
recorded in the draft's Implementation Record.

## The turn loop

A chat is pointed at a remote model per chat, in memory only
(`AppModel.remoteSelections`), so a relaunch returns every chat to local.
`executeGenerationTurn` routes such a chat to `executeRemoteGenerationTurn`
before it touches the local session. The remote turn uses the same lane
(`claimLane`), epoch, cancel flags and `finishTurnTail` as a local turn, and
hands its reply to `processCompletedTurn`, the post-generation tail extracted
from the local turn (state patch, dispatch gate, guardrail retry, prose or
tool-call tail). A remote tool call therefore meets the same dispatch gate,
the same permission ladder and the same ZP-23 background-parking rule as a
local one, and the existing step cap (`maxAutonomousSteps`) bounds a remote
agent loop. Tool definitions are sent natively. Images are not sent: the
request is text only, and the per-model Vision flag is recorded for a later
change.

A remote chat needs no loaded local model (`hasTurnModel`). The reply is built
into a `GenerationResult` through its decoder (timings zero) and tool calls
into `GenerationToolCall`, so nothing below the model call changed type.

## Context, compaction, usage

* The ring shows the provider's last reported prompt plus output tokens, or
  the labeled estimate (characters divided by 4) before one arrives.
* Auto-compaction uses the larger of the last reported count and the estimate
  against the model entry's window, with the local threshold. The summary runs
  on the chat's selected remote model (`performCompaction` branches on the
  chat). A provider context-overflow error is `contextOverflow` and its
  message offers compaction. Window fitting, recall and the instruction-pin
  cache are local-only and skipped for remote chats.
* `AppChatUsageLedger.byProvider` records remote turns by provider name, next
  to the existing totals. `AppChatUsageTotals.byProvider` sums it. No
  dashboard row is added.

## Rollout log

`logTrajectoryRemoteCall` writes a `modelCall` entry only when trajectory
logging is on and the chat is eligible (same gate as a local call). The entry
carries `remoteHost`, and its text passes the exact-key redaction and then the
store's own `TrajectoryRedactor` pass. The request body is never logged.

## Errors

Each `RemoteProviderError` maps to a localized, key-free message in
`AppModel.remoteErrorText`, shown in the same two places a local engine error
is: `AppModel.error` and an interrupted row (`finishCancelled(reason:
"error")`). No automatic retry.

## ATS and loopback http

The generated `Info.plist` (`scripts/make-app-bundle.sh`) has no
`NSAppTransportSecurity` key, so default ATS applies. https needs no
exception. Default ATS does not apply to connections made to an IP address, so
`http://127.0.0.1:PORT` and `http://[::1]:PORT` work, while a plain
`http://localhost` may be refused by ATS in the bundled app. Prefer the IP
form. Allowing `localhost` would mean adding `NSAllowsLocalNetworking`, a
release-script change that was not made here.
