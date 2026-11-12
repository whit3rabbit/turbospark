# Stage 7: Critic report (OC-02)

Round 1 only (no remediation, so no re-audit; convergence not assessed). Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). All completed. The first attempt to spawn structural and style hit "Concurrent subagent limit reached"; both were retried one at a time and succeeded. Standards: feature-spec-standard for design and structural.

## Design (DC), 11 findings (0 Blocker, 9 Major, 2 Nit)

DC-1 Major, lines 19-198. No Alternatives Considered section (mandatory). Missing: reuse the in-app toast path, dock bounce only, one master toggle, post even when focused, UNNotificationSound instead of NSSound. ZP-02 has one.
DC-2 Major, lines 54-55, 115-130. Click routing undefined: delegate ownership and lifetime (must be set before launch finishes or cold-launch clicks are lost, awkward with "no prompt at launch"), deleted or archived chat, willPresent behavior, ZP-23 interaction.
DC-3 Major, lines 56-59, 131-135. Authorization owner contradicts existing code: PushNotificationExecutor requests authorization at tool-call time (AutomationExecutors.swift line 51); "keeps its own use" vs "migrates" contradiction; state behavior when notDetermined, keep the "in-app only" fallback text accurate, add a test.
DC-4 Major, lines 48-50, 97-102. "NSApp not active" under-specified: settings window, occlusion, other Space, locked screen, Focus modes. Define "attention-eligible" (for example active, key window visible, not occluded) or justify isActive.
DC-5 Major, lines 51-53, 100-102. Dedupe inconsistent and unsafe: no event priority (finished or error can replace needsPermission); slot lifecycle undefined (refocus vs click vs "handled"); request and thread identifier scheme and removeDeliveredNotifications unspecified.
DC-6 Major, lines 26-29, 69-71, 79-84. Dependencies on unbuilt work: INDEX says OC-02 consumes ZP-02 state, but the diagram has no such link; desktop-shell-hardening brief link does not resolve in this worktree. Name the interface (for example ChatBadgeState), fix the link.
DC-7 Major, lines 79-84, 164-170. Hook points asserted, not verified: no function names; cancelled turn must not fire turnFinished; subagent and background-agent completion (BackgroundAgentNotification) and error-retry unaddressed; hook must carry chat id, not selected chat.
DC-8 Major, lines 56-59, 104-111. Degraded state incomplete and contradictory: authorization revoked later (refresh on activation), alert style None, unsigned or debug build with no bundle identifier (executor has a bundle guard, wrapper does not), and line 58 vs CUJ 3 toggle contradiction.
DC-9 Major, lines 60-62, 128. Sound model: NSSound follows output volume, not alert volume; bypasses Focus and Do Not Disturb; plays with authorization declined; set `content.sound` instead (decision belongs in Alternatives); sound list unenumerated, system sound names may change across macOS versions.
DC-10 Major, lines 137-146. No Risks and Limits section: notification floods across chats (suggest rate cap or "N chats need you" coalescing), chat titles and MCP tool names are sensitive on lock screen, privacy claim weaker than stated, no privacy test. (Overlaps DC-8.)
DC-11 Major, lines 139-140. 200 ms latency unjustified; `add` is async with no delivery guarantee. Define what is measured or drop.
DC-12 Nit, lines 174-176. Toggles persisted "with the archive"; ZP-02 uses MacAppSettings. No defaults or migration story.
DC-13 Nit, line 36. Broken link; needs ../../../swift/...; .kiro link has correct depth but missing target.
DC-14 Nit, lines 3-11, 31-42. OpenCode behaviors not re-verified; cross-window dedupe is meaningless in a single-window app.
DC-15 Nit, lines 64-71. No Non-Goal for scheduling or quiet hours though line 77 mentions scheduling.
DC-16 Nit, lines 75-78 vs 160-163, 179-198. Open question leans General but item 3 hard-commits; UNUserNotificationCenter is not directly mockable so a protocol seam must be listed; no acceptance criteria beyond CUJs; add Focus mode and locked-screen manual checks; status row accessibility labels.
DC-17 Nit, lines 79-84. Auto-resolve "yes" lean depends on ZP-02 option (b), itself open.

## Structural (ST), 13 findings (2 Blocker, 5 Major, 6 Nit)

ST-1 Blocker, whole document. No Alternatives Considered section (standard fails a single-path spec). Candidates: wrapper vs legacy NSUserNotification, unfocused-only vs always, NSSound vs UNNotificationSound, replace vs stack dedupe, policy object vs inline checks. No maintainer-attributed rejections without link.
ST-2 Blocker, whole document. No Risks and Mitigations section. Cover spam, stale dedupe slot (who clears on activation), revoked authorization, click on deleted chat, double sound, focus edge cases, coupling to four AppModel files.
ST-3 Major, lines 79-84. "Event source wiring" is a settled design decision inside Open Questions; move it to the architecture, keep only the countdown question; Open Questions precede the design, move later.
ST-4 Major, lines 131-135 vs 171. AutomationExecutors keep-vs-migrate contradiction; no statement how automation notifications fit unfocused-only and dedupe.
ST-5 Major, line 36. Broken relative link.
ST-6 Major, lines 116, 164. `AppAttentionEvents.notify` absent from the file list; "one call" vs four files; file-to-event mapping missing.
ST-7 Major, lines 26-29, 70-71. No Dependencies or Sequencing section; state what happens if ZP-02 slips; CUJ 2 hides the dependency.
ST-8 Nit, lines 56-59, 109-111. Degraded-state UI and Allow explainer have no file or view in the plan.
ST-9 Nit, lines 164-167. Four-path file item inconsistent with others; use sub-bullets.
ST-10 Nit, lines 59, 142-146, 198. Bare references (permission rules, announcements, DESIGN.md section 9); link them.
ST-11 Nit, lines 189-198. Verification lacks tests for AppModel hook emission and persistence.
ST-12 Nit, lines 3, 6. Section 3 and 9a references lack anchors.
ST-13 Nit, lines 202, 13-15. Restates ST-1 and ST-2 (no entries in doc's own contents).

## Style (SY), 17 findings (0 Blocker, 7 Major, 10 Nit)

SY-1 Major, lines 131-134 (and 171-173). Keep-vs-migrate self-contradiction.
SY-2 Major, lines 56-59 vs 36-37. Authorization claim vs executor requesting at tool-call time; "one owner" uncalibrated.
SY-3 Major, lines 56-58 vs 104-111. Declined-authorization toggle contradiction.
SY-4 Major, lines 33-38. "Nothing tells the user" overstated (automation executors post notifications); dock badge claim uncited.
SY-5 Major, lines 60-62, 128. Alert volume claim unverified; sound gating when authorization declined unstated.
SY-6 Major, lines 93, 140-142. "Chat title only" uncalibrated if titles derive from the first user message; CUJ 1 text has no tool name vs "names the tool".
SY-7 Major, lines 174-176. "Archive" vs MacAppSettings; name the actual store.
SY-8 Nit, lines 25-29, 70-71. "Shared state source" absent from the diagram; state ZP-02 sequencing.
SY-9 Nit, lines 38-42, 50. Unsourced OpenCode claims; "same shape fits ... without any cloud component" is filler.
SY-10 Nit, line 46. "Small set of system sounds" unmeasurable; name values.
SY-11 Nit, lines 48-49, 88. Terminology drift: not active, no window focused, unfocused, in background.
SY-12 Nit, lines 73-84. "18 panes" uncited; "scheduling" undefined; retitle "Event source wiring"; name decision owner.
SY-13 Nit, line 117. `AppAttentionEvents.notify` vs `AttentionEventPolicy` naming.
SY-14 Nit, lines 139-146. NFRs lack verification; VoiceOver mechanism uncited; in-window cards rely on ZP-02 pills "when built".
SY-15 Nit, lines 164-167. Error path assigned to AppModel+Cancellation.swift without evidence.
SY-16 Nit, structure. No Alternatives section (ZP-02 has one).
SY-17 Nit, line 36. Broken link.
