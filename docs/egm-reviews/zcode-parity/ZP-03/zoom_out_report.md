# Stage 5: Zoom out report, ZP-03 model trajectory viewer

Target: ZP-03-model-trajectory-viewer.md (full file, 185 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out (ZO)

Verdict: not ready to implement; no Blockers.

ZO-1 Major, lines 142-146, 94-98, 51-53. A bespoke JSONL store duplicates existing per-profile SQLCipher ProfileDatabase infrastructure (atomic appends, indexed tail, search, per-chat delete, caps). The design never says why a table is worse. Encrypting a JSONL stream and tailing it is non-trivial; ProfileArchiveStreams.swift and ProfileVault.swift are the nearest mechanisms and are not named.
ZO-2 Major, lines 51-53, 131, 138-146. Open Questions (vault vs sidecar, retention) are not closed but the plan commits. Full args/result duplicate the chat archive; decide storage and a per-entry payload cap.
ZO-3 Major, lines 33-34, 131-132. Ghost chats are not addressed. AppChatUsage.swift documents that ghost rows carry no conversation content; logging full args/results would break that invariant. State: no rollout for ghost chats.
ZO-4 Major, lines 33-36, 142-146. Log lifecycle unspecified: chat delete, branch/fork, edit-and-rerun, regenerate, rewind, archive, profile export/delete. Entries need a turn or message id link.
ZO-5 Major, lines 106-109, 29-30. Schema does not serve the stated problem (sampling artifacts, bad tool args): no sampling settings, no prompt hash, no native vs text-parsed tool call distinction. "Model call" scope undefined (foreground, subagents, title, goal loop, compaction).
ZO-6 Major, lines 77-83, 108-109, 113-120. Crash-safety weaker than claimed; failed or hung calls leave no record; CUJ 3 says "absent or marked truncated", two behaviors. Alternative A rejection is asserted. One small start-of-call line is not streaming I/O.
ZO-7 Major, lines 142-146, 85-104. "Engine callbacks already carry usage, latency, finish reason" is unverified. INDEX.md line 31 says the draft "adds Rust-side rollout logging"; the draft is Swift-only. Reconcile.
ZO-8 Nit, lines 69-75, 133-134. Search-all, paging and the 10,000-call scale are speculative; ship bounded tail plus search of loaded entries.
ZO-9 Nit, lines 39-40, 131-132, 145-146. Export redaction is a feature in disguise; check OpenProfileExport.swift and AppChatShareDocument.swift for an existing helper, or drop export from v1.
ZO-10 Nit, lines 106-109, 162-170. Altitude uneven: JSON schema and mutation recipes, but no layer ownership of the hook.
ZO-11 Nit, lines 157-161. "Existing inspector/dock routing" not shown to exist; consider an Inspector section instead of a new pane.
ZO-12 Nit, lines 3-6, 12. No stated dependency on ZP-23 (concurrent chats) or ZP-22 (remote providers).

## Inventive catalyst (IC)

IC-1 Major, lines 33-34, 77-83, 108-109, 113-119. Reliability vs streaming-path isolation is posed as either/or. "In-flight turn" loss conflicts with per-call appends (line 34 vs 109); quit flush does not cover a crash. Resolution: bounded in-memory queue drained by a single writer off the generation path, plus a "turn started" marker line; single writer removes the per-chat lock.
IC-2 Major, lines 51-53, 131-132. Encryption vs cheap tail reads treated as exclusive; seek-from-end does not work on a stream-encrypted file. Resolution: small independently encrypted segments, or a table in the encrypted profile database.
IC-3 Major, lines 39-40, 131-132, 180-181. Full payloads sit unredacted on disk, a second copy of the transcript; no deleted-chat behavior. Resolution: truncate large args/result at write time with a marker, expire with the chat, default size cap (a requirement, not an open question).
IC-4 Nit, lines 69-75, 133-134. Search-all is linear and decrypts everything; optional sidecar index of tool name, kind, finish reason.
IC-5 Gap, line 106. No turn or message id in the schema; "jump from transcript" impossible.
IC-6 Gap, line 106. Single `tokens` field; prompt, completion, cached needed.
IC-7 Gap, lines 103, 174. Same Rust-scope inconsistency as ZO-7 (`make swift-lib` listed for a "Swift-only" change).
IC-8 Gap. ZP-23 sequencing: hook must key by chat id under concurrent generation.
IC-9 Gap, lines 163, 169. No tests for concurrent appends or an unreadable segment.
