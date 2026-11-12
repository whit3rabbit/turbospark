# Stage 5: Initial zoom out report (OC-04 session tabs)

Target: docs/designs/opencode2-parity/OC-04-session-tabs.md (full file, 192 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed. IC could not see all of OPENCODE2_COMPARISON.md (truncated at 136 of 521 lines).

## Zoom out (ZO)

ZO-1 Major. Lines 43-49, 88-94, 114-116, 124-127, 151-154. Draft tabs duplicate existing `AppChat.draft`, debounced `promptText` writes and `materializeDraftChatIfNeeded()`; the model supports one transient draft (`selectedChatIndex == nil`), so several draft tabs need a multi-draft refactor the doc omits. Fix: make a draft tab point at a real chat row or the single `activeDraftChat`, drop the separate draft-text store, state the multi-draft decision.
ZO-2 Major. Lines 50-52, 84, 96-102, 114-116, 151-154. Closed-tab stack largely duplicates the chat list (closing never deletes a chat); only new data is strip position. Fix: v1 uses a short in-memory list of closed chat ids reopened via the existing select path; defer persisted stack and position restore.
ZO-3 Major. Lines 24-27, 53-55, 68-69, 81-83, 113-134. Ownership overlap with ZP-21 per-tab state and with split panes (`splitChatIDs`, `promoteSplitPane`); two persisted open-sets will drift; per-tab scroll has no cited mechanism. Fix: say whether session tabs live inside ZP-21's container; one rule for tab-versus-pane membership; cite where scroll lives.
ZO-4 Major. Lines 24-27, 138-139 (INDEX 76-78, 97). Without ZP-23, `selectChat`/`createChat`/`promoteSplitPane` refuse while generating, so tabs are weaker than the sidebar. Fix: hard prerequisite plus stated interim behavior on a refused switch.
ZO-5 Major. Lines 56-59, 171-172. Cmd+W is reserved in OC-01; last-tab behavior undefined; Option+Arrow fights composer word navigation; Cmd+Shift+[ and ] already cycle chats; catalog timing inconsistent. Fix: reuse Cmd+Shift+[ ], define last-tab Cmd+W, pick one catalog timing.
ZO-6 Major. Lines 131-134, 151-154. A second archive of draft text and chat ids skips `makeChatArchive`, the ghost-chat guarantee. Fix: ghost chats never appear in tabs, drafts or closed entries, with a test, or reuse the ghost filter.
ZO-7 Nit. Lines 138-139, 175-179. "Must not reload from disk" is vacuous (chats are resident); mutation-check wording is plan-level. Fix: delete the bullet, shorten the other.
ZO-8 Nit. Whole doc. No Alternatives Considered; cheapest option is an open-chat id list plus MRU in the sidebar. Fix: add Alternatives including that option.
ZO-9 Nit. Lines 32-33, 38-39, 75-80. Repo-root-relative Swift links do not resolve; "its report" lacks an antecedent; strip hidden at one tab hides the feature. Fix: use `../../../swift/...`, name the report, state discoverability.

## Inventive catalyst (IC)

IC-1 Blocker. Lines 56-58. Cmd+W is both OC-01 system-reserved and claimed as close tab; last, draft and Settings cases undefined. Fix: make it a fixed command, remove from the reserved set, define last-tab behavior.
IC-2 Major. Line 57 (and 165). Option+Left/Right collides with text editing; no reconciliation with existing prev/next chat shortcuts. Fix: use Cmd+Shift+[ ] and fold the old shortcuts in.
IC-3 Major. Lines 124-125 vs ZP-21 98-101. Three writers of "what is selected" (ZP-21 slice, new store `selectedTabID`, `AppChatFileStore.selectedChatID`). Fix: session tab set is a field of ZP-21's slice; the chat archive owns `selectedChatID`.
IC-4 Major. Lines 46-49, 124-128. Draft tab kind, second draft store and promotion swap duplicate `materializeDraftChatIfNeeded` and `AppChat.draft`. Fix: draft tab is a lazily materialized empty chat row; needs GC and sidebar filtering rules.
IC-5 Major. Lines 46-47, 131-135. Plaintext draft store sits outside the ghost guarantee and adds per-keystroke writes. Fix: exclude ghosts, debounce like `persistChatsDebounced`.
IC-6 Major. Lines 161-162. Shutdown flush and restore order unspecified (`shutdown()` flushes only chats, settings, file store; restore needs `loadChats` and `loadProjects`; project mismatch). Fix: explicit tab flush, ordered restore, ID validation.
IC-7 Major. Lines 50-52, 129. Closing a busy, approval-pending or dirty-draft tab undefined; draft close contradicts "never loses an unsent prompt". Fix: running tab keeps generating; draft with text is confirmed or pushed to the stack; state behavior without ZP-23.
IC-8 Major. Lines 79-80, 119-121. Strip hides at one tab plus one draft, so Cmd+T shows nothing; "never truncate" conflicts with width caps and pane-header precedent. Fix: show at two or more tabs; minimum width, tooltip, full title in a11y label and overflow menu.
IC-9 Major. Line 84 vs 113-116. Closed stack called session-scoped but stored in the persisted store. Fix: pick in-memory only and say relaunch clears it.
IC-10 Major. Lines 98, 155-157. Chats deleted while a tab or closed entry exists are not handled; dropped restored tabs are silent. Fix: observe chat deletion, prune, toast once.
IC-11 Major. Lines 68-69, 81-83. Same chat can be a tab and a split pane; promote/close interplay undefined. Fix: a chat is in at most one tab; pane list excludes the selected tab's chat; promote selects a tab.
IC-12 Nit. Lines 52-54 vs 114-116. Per-tab scroll has no storage. Fix: in-memory, not restored, or add to schema.
IC-13 Nit. Lines 48-49, 127-128. Failed promotion on submit refusal not covered. Fix: swap kind only after submit accepts.
IC-14 Nit. Lines 58-59 vs 171-172. Chord registration wording inconsistent with INDEX 79-80. Fix: add entries with stable ids; exposure waits for OC-01.
IC-15 Nit. Lines 32-33. Broken repo-relative links. Fix: `../../../swift/...`.
IC-16 Nit. Lines 175-188. Verification omits ghost guarantee, delete-while-open, close-running, shutdown flush. Fix: add four cases with mutation checks.
