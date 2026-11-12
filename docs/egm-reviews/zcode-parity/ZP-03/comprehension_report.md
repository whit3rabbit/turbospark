# Stage 6: Comprehension report, ZP-03

Reviewer: egm-comprehension (CO), read the full file alone. Completed.

## What this proposes

Add a per-chat trajectory pane to the TurboSpark macOS app: a role-separated timeline of every model call and tool execution in a chat. Each entry shows model name, tokens, latency and finish reason, and expands to arguments, results and errors, with search and payload inspection. Each chat gets a JSONL rollout log; one line is appended after each model call or tool execution finishes, never during streaming, under a per-chat lock. The pane reads only the last 500 entries, with "load earlier" paging and an off-main-thread "search all". Export is redacted; the on-disk file stays unredacted.

## How the existing system works, per this document

No per-call record exists. There is a transcript row, aggregate diagnostics in "the Inspector", and a per-chat usage ledger of aggregates. Debugging today means reproducing on the CLI with TURBOSPARK_DEBUG_PROMPT_IDS. The doc says engine callbacks already supply usage, latency and finish reason, a "profile vault" gives encryption at rest, and "inspector/dock routing" exists in ChatTranscriptView. None of these is described beyond one-line mentions.

## Findings

CO-1 Major, lines 39-40, 51-53, 131-132. Privacy model contradicts itself and relies on an undefined "profile vault"; location is open (vault vs sidecar) yet NFR states vault encryption as settled; unclear whether tail reads work on an encrypted file.
CO-2 Major, lines 83, 108-109, 117-119. Crash-safety does not connect: what the quit flush flushes under append-after, whether "after the fact" means per call or per turn, what a hard crash loses, whether "marked truncated" is a real state.
CO-3 Major, lines 106-107. Schema too thin: `tokens` not broken down, `role` values unlisted, toolExec field population unstated, no turn id, `v`/`ts` formats missing, "model call" undefined.
CO-4 Major, lines 14, 19, 51-52, 142. Rollout identity unclear: one log per chat vs `model-io-*.jsonl`; rotation; behavior on chat delete, rename, archive; "chat archive" undefined.
CO-5 Major, lines 38, 69-75, 96-98. Bounded tail specified by count, mechanism not (reverse scan or index), no byte bound, no handling of huge payloads.
CO-6 Major, lines 39-40, 132. Redaction under-specified: "user-writable paths" and "secret-shaped strings" undefined, best-effort or not, "shareable" uncaveated.
CO-7 Major, lines 49-55. Open Questions gate the data model and privacy claim but the doc commits elsewhere; no decision owner or date; no retention tests.
CO-8 Nit, lines 3-6, 28, 99-104. Undefined project terms (Inspector, usage ledger, engine callbacks, dock routing, DESIGN.md, excluded category, model gates, catalog strings).
CO-9 Nit, lines 28, 174-178. Unexplained commands and counts (`make swift-lib`, `make design-check`, "ten parity gates", "21 languages", TURBOSPARK_DEBUG_PROMPT_IDS).
CO-10 Nit, lines 135-136. "No model gates apply" assumes reader knowledge and is asserted, not checked.
CO-11 Nit, lines 89-104 vs 142-153. Diagram and file list disagree (AppModel+Trajectory not in diagram; Diagnostics/ vs Generation/ placement unexplained).
CO-12 Nit, lines 13-21, 59-75. Default view unstated: whole chat (CUJ 2) or selected turn (CUJ 1).
