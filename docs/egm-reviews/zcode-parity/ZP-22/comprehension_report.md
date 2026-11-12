# Comprehension report: ZP-22 remote providers (Stage 6)

Reviewer: egm-comprehension (persona agents/comprehension.md). Read the document alone, no standard, no context files.

## What this proposes

Add opt-in remote model providers (Anthropic and OpenAI-compatible APIs) to the TurboSpark Swift app, using user-supplied API keys stored in Keychain. Users configure providers and model lists in a Providers settings section. Remote models appear in a separate, labeled Remote section of the composer model picker. Local weights stay the default. Remote chats reuse the existing turn loop, tool permission ladder, usage ledger, context ring and rollout log. There is no OAuth, no hosted service and no automatic failover. A new Swift-side step abstraction (ModelStep) sits behind the turn loop, with a local implementation wrapping the existing FFI path and a remote one calling per-family SSE clients.

## How the existing system works, per this document

Chat runs only on local weights through an FFI path (Swift calling Rust). A Swift "turn loop" orchestrates each step. It already parses native tool calls for the local engine, and it handles steering, queues, stop and compaction. Tool calls go through a "tool registry" and a "permission ladder". Usage is tracked in a per-chat usage ledger and a context ring. Settings are persisted as JSON through AppJSONStore. Secrets follow existing "credential rules", with per-credential opt-in Keychain items. The Rust server already speaks both API families, but only in the serving direction. Whether the turn loop is actually independent of FFI at the "model step" seam is unverified, and the document lists it as an open question.

## Findings

CO-1 Blocker, lines 69-75, 138-149, 214-219. The design depends on a seam the document says may not exist; the reader cannot tell whether the real plan is chat-only or chat plus tools. Fix: verify the seam before approval and record the result, or write both landing plans with a decision rule.

CO-2 Blocker, lines 93-96 (CUJ 1). "A 'store in Keychain' toggle defaults on and off means the provider stays disabled" cannot be paraphrased with confidence and contradicts line 52 ("per-credential opt-in"). Fix: separate sentences giving the default, the off behavior, and where the key lives when off.

CO-3 Major, lines 52, 94, 120-124, 202-205. "Repo rules", "existing credential rules" and "never at launch" are never stated or linked. Fix: define inline in two or three bullets or link the doc.

CO-4 Major, lines 114, 140, 232-234. Undefined project terms: permission ladder, tool registry, step events, usage watermarks, pre-send ring, expert residency, window fitting, ledger. Fix: one-line definition at first use, with links.

CO-5 Major, lines 3, 29. Findings 7.1, 7.2 (API-key half), 7.3/7.4 are cited without saying what they are. Fix: one-line summary of each.

CO-6 Major, lines 214, 220. Modify targets given as "basenames" for unnamed files. Fix: list exact paths.

CO-7 Major, lines 52-53, 183-186. "Key-shaped strings" undefined, no pattern source; first-send notice does not say whether it blocks or is a banner, per chat or per provider. Fix: specify pattern source and notice behavior.

CO-8 Major, lines 191-193. Error taxonomy not mapped to the "local engine error handling" it claims to match. Fix: state the existing error model and the mapping.

CO-9 Major, lines 76-80 vs 48-51 and CUJ 2. Context ring option (a) is "lean" in Open Questions but settled in Goals and CUJ 2; the "documented local estimate" is not documented. Fix: close the question and state the chars-per-token value, or mark the goal provisional.

CO-10 Nit, lines 21, 57-58. "No provider family switching" is ambiguous. Fix: clarify or delete.

CO-11 Nit, line 122. CUJ 4 says "exports settings" but no export feature is described. Fix: name the feature or drop the word.

CO-12 Nit, line 148. ZP-03 referenced without link or description; "remote marker" format unspecified. Fix: link ZP-03 and state the marker field.

CO-13 Nit, lines 194, 256-258. "Ten parity gates", "21 languages", `make design-check` unexplained. Fix: link or name them.

CO-14 Nit, lines 188-190. Timeouts and stall detection have no values or user-visible outcome. Fix: give defaults and outcome.

CO-15 Nit, lines 26-32. The sentence saying server code "proves" the codebase speaks both families is tangled; the Rust code serves rather than calls these APIs. Fix: split and state that.

CO-16 Nit, lines 81-85. Test connection probe unsettled yet CUJ 1 promises "a named failure". Fix: list failure names or reference the error taxonomy.
