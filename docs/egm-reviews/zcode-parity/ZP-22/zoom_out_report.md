# Zoom out report: ZP-22 remote providers (Stage 5)

Target: docs/designs/zcode-parity/ZP-22-remote-providers.md, full file.

## Zoom-out reviewer (egm-zoom-out, persona agents/zoom_out.md, standard zoom-out-standard.md)

Verdict: context sufficient, root problem clear, design not ready. 1 Blocker, 5 Major, 4 Nit. Reviewer verified against the worktree: no Networking/ directory, no ModelStep, no existing /v1/messages or /chat/completions client; AppModel+Generation.swift builds a local `[GenerationToolCall]` at line 302; keychain wrappers exist in Tools/Hooks/HookOptionKeychain.swift, Server/ServerKeychain.swift, Server/TypeSafeKeychain.swift; URLSession is in Tools/Web/HttpRequestExecutor.swift and WebFetchExecutor.swift. Caveat: ZCODE_COMPARISON 7.1/7.2 text was not read; finding 1 is an inference from line 302, not a full loop read.

ZO-1 Blocker, lines 67-75, 138-149, 214-219. Architecture rests on a ModelStep seam that does not exist; verification is deferred to "the first implementation task" while the doc already commits to extracting a protocol and to LocalModelStep wrapping FFI "unchanged". Loop uses local `GenerationToolCall`. Fix: do the seam read now, name the real call site and event types, then commit to the extraction or cut tools from scope.

ZO-2 Major, lines 20-23, 33-53, 126-149, 197-250. Scope bundles Settings, two clients, a loop refactor, context accounting, ledger and rollout log in one landing; no phasing. Fix: phase 1 is OpenAI-compatible chat-only with a flat model list; Anthropic and tools later.

ZO-3 Major, lines 206-213. Two bespoke SSE clients in a new Networking/ folder with no shared SSE parser or transport; ignores existing URLSession code in Tools/Web. Fix: add a shared SSE stream and transport layer, state what is reused, or place files in an existing directory.

ZO-4 Major, lines 53, 92-96, 246-250. "Existing credential rules" unnamed; three keychain wrappers exist. CUJ 1 contradicts itself: opt-in "satisfied by entering", then a default-on "store in Keychain" toggle whose off state disables the provider. Fix: name the helper to extend and drop the toggle; saving the key is the consent act.

ZO-5 Major, lines 46-51, 76-80, 151, 231-234. Context accounting left open while the plan already edits AppContextUsage; "compaction works unchanged" is likely false because compaction is driven by the local tokenizer count. Fix: decide (a), specify the remote compaction trigger source and the provider overflow error path.

ZO-6 Major, lines 31, 108-111, 148. Rollout log (ZP-03) is a draft with Rust-side logging; INDEX says ZP-22 "stands alone". Fix: make it an explicit soft dependency or drop it from CUJ 2; update INDEX.

ZO-7 Nit, lines 214, 220. Paths not real: "basenames", `App/SettingsPanes.swift` not found; generation files are State/AppModel+Generation.swift and State/AppModel+AgentLoop.swift. Fix: use real paths.

ZO-8 Nit, lines 81-84, 189-193, 243-244. Connection check is stated as a decision ("Include") yet listed as an open question; NFR and tests carry implementation detail (mutation checks, per-family taxonomy). Fix: move Test connection to Goals; move detail to the implementation spec.

ZO-9 Nit, lines 164-172. Alternative B rejection cites "FFI/wire/binding cost" with no size comparison to two Swift clients. Fix: add a short size comparison or say why serving-side code is not reusable.

ZO-10 Nit, lines 17-22, 194. Gating rule stated only in tests; "21 languages" vs "ten parity gates" unreconciled. Fix: state the gating rule once in Goals; cite the verification doc for gate counts.

## Inventive catalyst reviewer (egm-inventive-catalyst, persona agents/inventive_catalyst.md)

Caveat from reviewer: read ZCODE_COMPARISON.md lines 1-237 only (7.1/7.2 unread); repo paths and turn loop not checked. Persona format is Major or Nit only.

IC-1 Major, lines 45-47, 69-75, 109-115, 214-219, 241-242. Tool loop is an open question yet Goals, CUJ 3 and tests assume it lands together. Fix: add a Phase 0, seam extraction with LocalModelStep only and a ladder-equality test, before any client; gate Goals and CUJ 3 on it. (Would be Blocker-class if the seam check fails.)

IC-2 Major, lines 52-53, 119-124, 183-187. Arbitrary base URL with only log redaction: redirects, http:// base URLs and custom headers uncovered, so the key can reach another host. Fix: one egress type that injects Authorization last, holds the key in a non-Codable non-printable wrapper, strips credentials on cross-host redirects, refuses non-HTTPS except loopback; add a redirect-strips-key test.

IC-3 Major, lines 64-65, 151-153, 183-185. "Explicit selection" covers only the user's send; compaction, titles, memory, goal timers, cron prompts and a mid-chat switch to remote can send history or vault attachments without a choice; notice shows only at first send. Fix: add a "Remote egress scope" subsection listing each background caller and its default (local-pinned, remote compaction per-provider opt-in), plus a per-chat consent shown before the first remote send.

IC-4 Major, lines 49-51, 76-80, 231-234. Context accounting framed as heuristic vs per-family tokenizers; a third option exists. Fix: replace (a)/(b) with a watermark-calibrated chars-to-token ratio plus an optional count endpoint where the family has one.

IC-5 Major, lines 188-193. Error taxonomy has no mitigation: no Retry-After, backoff or partial-stream retention, unlike the local path. Fix: add bounded backoff before first token, interrupted row with Continue mid-stream, manual "Retry on local model", and no auto-retry for tool-bearing turns.

IC-6 Major, lines 35-38, 42-44, 204-213. "OpenAI-compatible" treated as one protocol; reasoning fields, tool_calls deltas, usage, image encoding and cache fields differ. Fix: add a Capability mapping section with a per-provider quirk profile feeding the one step-event contract.

IC-7 Nit, lines 93-96. Opt-in toggle contradictory. Fix: define the off state as a session-only in-memory key.

IC-8 Nit, Non-Goals. Silent on whether the loopback server may route to remote models (ZCODE_COMPARISON 1.4). Fix: state the local server never routes to remote providers.

IC-9 Nit, lines 188-189 vs INDEX 89-105. Interaction with ZP-23 unstated; INDEX says "stands alone" but sequences after ZP-23. Fix: state ZP-22 neither depends on nor blocks ZP-23 and align INDEX wording.

IC-10 Nit, lines 81-84, 214, 220. Files named by "basename"; tiny-completion probe may cost money. Fix: name exact files after the seam check; prefer a free models-list probe.
