# Stage 7: Critic report (OC-04 session tabs)

Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY), one round, direct dispatch. All completed. Standard: feature-spec-standard for DC and ST. Convergence not assessed (one round, no remediation).

## Design critic (DC)

DC-1 Blocker. Whole file, 73-84. No Alternatives Considered and no Risks & Mitigations; missing alternatives: split panes only, recent-chats menu, drafts on `AppChat.draft`. Fix: add both sections.
DC-2 Blocker. 46-47, 114, 131-134. Persisted draft text bypasses `makeChatArchive` ghost filter; ghost session ids and plaintext prompts reach disk; vault lock and profile switch unspecified. Fix: never persist ghost tabs or drafts, reload on profile switch and vault lock, add tests.
DC-3 Major. 46-47, 107-109, 131-134. Two sources of truth for session drafts (`AppChat.draft` vs "ids only" vs CUJ 3); `materializeDraftChatIfNeeded` not reconciled. Fix: session drafts stay on `AppChat.draft`; store holds draft text for draft tabs only.
DC-4 Major. 113-129, 161-162. `selectedTabID` vs `selectedChatID` can disagree; restore order, first-launch migration and schema version missing. Fix: derive `selectedChatID` from the tab or define reconcile, plus migration and version.
DC-5 Major. 46-53, 138-139. Per-tab scroll claimed with no storage; the no-reload NFR is vacuous (all chats resident). Fix: say where scroll lives; measurable view-rebuild requirement.
DC-6 Major. 56-59. Cmd+W reserved in OC-01; Option+Arrow is composer word-move; last-tab close would quit the app; reopen has no chord though file 7 says three chords. Fix: non-conflicting chords, last-tab behavior, reopen chord.
DC-7 Major. 31-37, 68-69, 81-83. Tab and split-pane ownership undefined (same chat in tab and pane, promote, duplicate tabs, promote disabled while running). Fix: one tab per chat id and a pane/tab ownership rule.
DC-8 Major. 24-28, 98-103, 125-129. Layering with ZP-21 and ZP-23 hand-waved: owner of per-tab slice, workspace scoping, cross-project reopen, strip placement conflict with ZP-21, generating-chat close, degraded mode without ZP-23. Fix: define each.
DC-9 Major. 84, 116, 143-146, 151-154. Closed stack "bounded 20" vs lean, session-scoped vs persisted; reopen after in-session deletion not covered. Fix: decide persistence, validate at reopen.
DC-10 Major. 47-49, 88-94, 127-129. Promotion failure, atomicity, no-model and pre-ZP-23 guard behavior unspecified; "no flash" has no mechanism. Fix: swap kind only after the chat row exists; describe failure.
DC-11 Nit. 77-80, 119-121. Hide rule ambiguous (hides a fresh Cmd+T draft); "never truncate" vs caps; no tab-count cap. Fix: precise rule and cap.
DC-12 Nit. 7-10, 32-33, 78-79. Unverified upstream claims; broken `swift/...` links. Fix: note status, use `../../../swift/...`.
DC-13 Nit. 175-179. Tests omit ghost/vault, selection reconcile, duplicate-tab invariant, promotion failure. Fix: add with mutation checks.

## Structural (ST)

ST-1 Blocker. Missing Alternatives Considered (standard's Alternatives Rule; siblings have it). Fix: three alternatives with pros, cons, concrete rejection reasons.
ST-2 Blocker. Missing Risks & Mitigations (plaintext drafts, deleted chats, tab-count limits, Cmd+W change vs OC-01 reserved list). Fix: add the section.
ST-3 Major. 56-59, 167-172. OC-01 dependency and ordering unstated; Cmd+W vs reserved list unreconciled. Fix: Dependencies line naming ZP-21, ZP-23, OC-01 and ordering.
ST-4 Major. 53-55, 24-28, 111-134. Second store with no relation to ZP-21's per-tab slice; draft ownership unclear. Fix: show SessionTabStore under the WorkspaceTab slice.
ST-5 Major. 73-84 vs 41-60, 113-129. Placement, split-pane and size questions are "lean" in Open Questions but committed elsewhere. Fix: resolve or mark provisional.
ST-6 Major. 147-179. File entries are Rationale-only unlike sibling template; item 5 silent on previous/next chat with a draft tab selected. Fix: pick one format; state draft-tab behavior.
ST-7 Major. 136-145, 181-188. No Persistence safety or Resource bound NFR; no CUJ for deleted-chat restore; localization gates not named. Fix: add them.
ST-8 Nit. 3-10, 32-34. Unlabeled link block; broken Swift links; "Section 9a" needs the other doc. Fix: `../../../swift/...`.
ST-9 Nit. 56-59, 131. Cmd/option capitalization; strip naming drift. Fix: unify.
ST-10 Nit. 4-10. Upstream links point to mutable dev branch; INDEX pins 7ce35879b8. Fix: pin.

## Style (SY)

SY-1 Major. 38-39. Unverifiable "ranks this gap fourth". Fix: cite the section or delete.
SY-2 Major. 36-39. "Whole model ... is absent" overclaims given split panes. Fix: "not found in the files read", name them.
SY-3 Nit. 20-22, 28, 38. Filler ("browser-style", "worth having"). Fix: measurable purpose.
SY-4 Nit. 49-50, 94. "No flash" not measurable. Fix: define the test.
SY-5 Nit. 68-69, 75-80. Open Questions duplicate Non-Goals; hide rule ambiguous; no decision owner. Fix: one statement, owner, closing point.
SY-6 Nit. 79, 120-121, 142-145. Absolute "never truncate"; caps have no values. Fix: numeric caps or tokens; say wrap vs overflow.
SY-7 Nit. 56-59, 171-172. Unsourced "macOS convention"; "option" capitalization; catalog sequencing contradiction. Fix: one sequencing rule; cite or drop.
SY-8 Nit. 183, 186. Scheme not named; DESIGN.md section 9 unlinked. Fix: name and link.
