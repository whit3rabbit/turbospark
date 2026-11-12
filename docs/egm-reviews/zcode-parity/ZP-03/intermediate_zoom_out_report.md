# Stage 8: Intermediate zoom-out report, ZP-03

Reviewer: egm-zoom-out (S8). Completed. Document unchanged since Stage 5, so every earlier finding is open. Verdict: not ready; the only Blocker is the missing Risks section (ST-1).

Root cause: a hand-built per-chat encrypted append-only tail-readable JSONL store drives about 20 findings. A table in the existing SQLCipher ProfileDatabase (State/ProfileDatabase*.swift; ProfileVault.swift, ProfileVaultCrypto.swift, ProfileArchiveStreams.swift, AppChatUsage.swift exist and are unnamed in the draft) gives transactions, indexed tail, delete-with-chat and cap-by-trim.

S8-M1 Major, lines 51-53, 94-98, 111-127, 142-146. Replace RolloutStore with a ProfileDatabase table; rewrite Alternative B as "table vs file". [ZO-1, DC-5, ST-3, IC-2]
S8-M2 Major, lines 77-83, 108-109, 118-119. Rewrite CUJ 3: completed calls present, in-flight call lost; drop flush and "truncated". Decline IC-1 queue and marker. [SY-1, IC-1, CO-2, DC-3, ZO-6]
S8-M3 Major, lines 33-34, 131-132. One Goal line: no rollout for ghost chats. [ZO-3]
S8-M4 Major, lines 33-36, 142-146. Cascade delete with chat; fixed per-payload truncation cap; cap plus per-chat clear as the only controls; close the Open Questions. [ZO-4, CO-4, DC-2, IC-3]
S8-M5 Major, lines 99-104, 174, INDEX l.31. Name the callback seam or say it does not exist; fix INDEX. [ZO-7, DC-1, ST-4, SY-2, SY-3, IC-7]
S8-M6 Major, lines 106-109. Minimal schema: turn id, token split, explicit kind; defer prompt hash, sampling, native-vs-text; define "model call" as foreground only; drop `v` with a table. Decline full expansion. [ZO-5, CO-3, DC-7, ST-7, IC-5, IC-6]
S8-M7 Major, lines 3-12, 138-152. One dependency line on ZP-23 (key by chat id). [DC-6, IC-8, ZO-12]
S8-B1 Blocker, missing section. Short Risks and Mitigations (4 to 6 lines). [ST-1]
S8-N1 Nit, lines 38, 69-75, 133-134. Cut search-all and off-main scan; tail plus search of loaded entries. [ZO-8, CO-5, DC-4, SY-9]
S8-N2 Nit, lines 39-40, 64-66, 132, 145-146. Drop export from v1 or reuse an existing redaction helper; drop "shareable". [ZO-9, CO-6, DC-11, ST-8, SY-6]
S8-N3 Nit, lines 89-104, 142-161. One placement (prefer Inspector section); reconcile diagram and file list; store under State/. [ZO-11, ST-6, CO-11]
S8-N4 Nit, lines 162-170. Drop mutation recipes if file store is deleted; name hook owner. [ZO-10, DC-10, IC-9]
S8-N5 Nit. Cite or verify TURBOSPARK_DEBUG_PROMPT_IDS, Inspector claim, ZCode timeline source, "ten gates", "21 languages". [DC-9, ST-12, SY-4, CO-8, CO-9, SY-11, DC-12, ST-5]

Decline (scope growth): IC-1 queue/writer, IC-2 segmented files, IC-4 sidecar index, full schema expansion, scale/concurrency tests, DC-13 catalog listing, ST-9 numeric budgets.
