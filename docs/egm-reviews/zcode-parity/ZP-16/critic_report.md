# Critic report (Stage 7): ZP-16 command center palette

One round, no remediation, convergence not assessed. Standard: feature-spec-standard for design and structural; style had no standard. All three reviewers completed. IDs: DC (design), ST (structural), SY (style).

## Design (DC)

* DC-1 Major, 35, 142-145, 175. "Three scope tabs" lists four; line 175 says "all three scopes". Fix: four tabs, three provider scopes.
* DC-2 Major, 111-126, 117. Both rejections are "Rejected by author", unlinked; "ZCode merged them for a reason" unsupported. Fix: concrete trade-offs; add real alternatives (chat-only Cmd+K plus separate palette, no Files scope, ranking variants).
* DC-3 Major, 19-22. Kiro spec `chat-runtime-qol` not found under .kiro/specs. Fix: link the real spec or drop the claim.
* DC-4 Major, 44, 156-159. Cmd+K is bound in TurboSparkApp.swift:463 (posts `.showChatSearch`); RootView.swift:132-135 hosts the overlay; neither is in Target Files; Cmd+Shift+K is Clear Chat History; "pre-filtered to commands" has no mechanism. Fix: add both files, specify the pre-filter mechanism.
* DC-5 Major, 38-40, 96-102. Command provider has no data source: menu actions are SwiftUI Buttons with no registry; enablement (menu disabled states) undefined; run versus insert-into-composer undefined. Fix: name a registry or action table, define enablement and dispatch.
* DC-6 Major, 41-42, 70-76, 132-134. Index caps (2000, depth 8, 20 s) truncate large repos; entries include directories; `entries(for:)` walks on cold cache, contradicting the 100 ms budget; no project means empty scope. Fix: add limits section, use cachedEntries on open, scope or drop the 100 ms claim.
* DC-7 Major, 70-76. `@file:` handoff edge cases undefined: composer unmounted or disabled, directory chosen, attach failure. Fix: define each.
* DC-8 Major, whole doc. No Risks and Mitigations section (overlay regression, execution bypassing menu enablement, stale history, history privacy). Fix: add it.
* DC-9 Major, 43, 169. History has no store, key, schema, dedup, clear affordance or privacy treatment. Fix: specify.
* DC-10 Major, 82-85, 107-109. All-scope merge and per-keystroke cost undefined; no debounce or threading. Fix: grouped sections with top-N each; state debounce and threading.
* DC-11 Major, 36-37. Prefix grammar undefined (space after prefix, literal `@`/`#`, deleting prefix, collision with `ChatSearch.Query.parse`). Fix: add short grammar.
* DC-12 Nit, 54-58. Home view has no default. Fix: state default (c), mark non-blocking.
* DC-13 Nit, 29-31. ZCode tabs are All, Actions, Tasks, Files; the draft renames silently. Fix: say so.
* DC-14 Nit, 165-166. Only two mutation checks. Fix: add chat-row parity, mention payload, plugin namespace.
* DC-15 Nit, 130-136. Accessibility generic. Fix: name tab-key scope switching, Esc, wraparound, focus return; plugin titles untranslated.
* DC-16 Nit, 174. Reference the ten localization parity gates.
* DC-17 Nit, 168-177. No release-evidence gate. Fix: link .claude/docs/verification.md.

## Structural (ST)

* ST-1 Blocker, no heading after 128. No Risks and Mitigations section required by the standard. Fix: add H2 covering plugin name collisions, index staleness and cap, history leakage, overlay regression.
* ST-2 Blocker, 113-126. Alternatives rejected "by author" with no link or technical reason. Fix: restate technically or link the decision record.
* ST-3 Major, 33-36. Four tabs versus "three". Fix: "four scope tabs (All plus three provider scopes)".
* ST-4 Major, 54-58. Home view open while Draft; history UI unspecified. Fix: state default as a goal or mark as approval blocker.
* ST-5 Major, 21-22. Kiro spec and author decision unlinked (INDEX Provenance 131-132 not linked). Fix: link both.
* ST-6 Major, 33-36, 44-45, 160-166. Plugin commands and Cmd+Shift+P have no test cases. Fix: add.
* ST-7 Major, 26. "Roughly 40" unsourced. Fix: cite registry count.
* ST-8 Major, 168-177. No check for the 100 ms claim or VoiceOver NFR. Fix: add measurement and accessibility audit steps.
* ST-9 Major, 64-85. CUJs omit Cmd+Shift+P, history reuse, and empty or failure states. Fix: add CUJ 4 and an empty-state CUJ.
* ST-10 Nit, 60-138. Section order versus template unconfirmed. Fix: compare to EGM template.
* ST-11 Nit, 3-7. ZCode links use `blob/main`; provenance docs not linked. Fix: pin to `a60ba6a7`, add docs.
* ST-12 Nit, 165-166. Mutation bullet mis-nested. Fix: move to Verification Plan.
* ST-13 Nit, 152-155. Modify versus replace/delete inconsistent. Fix: choose one.
* ST-14 Nit, 174. "21 languages" versus "ten gates". Fix: cite gate names.

## Style (SY)

No Blockers.

* SY-1 Major, 35-36. Tab count (see DC-1).
* SY-2 Major, 26-27. "Roughly 40" hedged count. Fix: exact count or drop.
* SY-3 Major, 136, 174. "21 languages" versus INDEX 116. Fix: one phrasing or cite gate.
* SY-4 Major, 107-109. "No new scoring claims" is meta-commentary; All ranking undefined. Fix: state interleave order, delete parenthetical.
* SY-5 Major, 54-58. Home view has no consequence and no goal line; "Author decides" unspecific. Fix: add goal line and name the decision.
* SY-6 Major, 43-44. Cmd+Shift+P initial state unspecified (prefix shown or tab selected). Fix: one sentence.
* SY-7 Major, 117-118, 125. Slogans instead of rationale. Fix: concrete reasons.
* SY-8 Nit, 132-134. 100 ms has no measurement method. Fix: name it or mark as target.
* SY-9 Nit, 119, 126. "Reason for Rejection: Rejected by author" repeats. Fix: keep the reason only.
* SY-10 Nit, 82-85. "plain-Enter" odd; replacement requirement inside a Then. Fix: reword, move to Goals.
* SY-11 Nit, 49-50. Non-goal not proposed; "recency-ordered match surfacing" undefined. Fix: "Chat search ranking is unchanged (most recent match first)".
* SY-12 Nit, 17-19. Wordy. Fix: "Chat search becomes one scope of the palette."
* SY-13 Nit, 165-166. Mis-nested (see ST-12).
* SY-14 Nit, 33-45. History store unnamed. Fix: name it.
* SY-15 Nit, 140-166. File list omits history store and plugin source. Fix: add or state existing store.
