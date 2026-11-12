# Zoom out report (Stage 5): OC-05 back and forward history

Target: docs/designs/opencode2-parity/OC-05-back-forward-history.md (full file, 132 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom Out reviewer (ZO)

Verdict: not ready as written. Context sufficiency passes. Two Blockers come from checking the cited Swift sources.

ZO-1 BLOCKER. Lines 38-39, 112-114. Cmd+[ and Cmd+] are already bound. `KeyboardShortcutCatalog.swift` lines 81-88 list "Previous Chat" as Cmd+[ and "Next Chat" as Cmd+], with Shift+Cmd alternates. The draft never mentions this. Shipping both gives two menu items on one chord or silently breaks chat cycling. A decision is needed: remap chat cycling, pick other chords, or make the chord context-sensitive. The catalog is the single source for the menu, the shortcuts pane and `KeyboardShortcutCatalogTests`, so this changes a contract. It also needs a note on OC-01 and OC-04.

ZO-2 BLOCKER. Lines 26-28, 77, 106-108. Wrong ownership of the recording point. `activeSection` lives on AppModel and is assigned from many places: NavigationRailView (lines 99-137), TurboSparkApp.swift line 327, AlternateShortcutBridge line 46, TopBarView, PromptContextPills, ModelLoaderControl, ModelManagerView, FilesSectionView, and AppModel+LoadGate lines 107, 111, 193. RootView only observes it (`.onChange` near line 228). The design must say history is owned by AppModel (a funnel or an observer) and how Back/Forward restores avoid re-recording themselves. Chat identity and settings tab id live in other model state, which is not named. Programmatic switches (LoadGate forcing chat) are not user jumps; the draft must say whether they push. Behavior when a recorded chat was deleted is undefined.

ZO-3 MAJOR. Lines 18-19, 36, 55-68. "Settings pane" and "search jump" are not verified as section-level. No settings tab id was found among section assignments. The Non-Goals exclude in-pane transitions while Goals include settings pane opens. Name the real types and state. The collapse rule needs a precise definition (previous entry only or any settings entry; the entry keeps the latest tab).

ZO-4 MAJOR. Lines 38-39, 80-83. The View menu group at TurboSparkApp.swift line 324 is `CommandGroup(before: .toolbar)` derived from `AppNavigationSection`. The draft does not say where its group goes. "AlternateShortcutBridge: no extra chords needed" is unsupported: the bridge already rebinds section chords (line 46).

ZO-5 MAJOR. Line 44. Menu items do not show tooltips on macOS. The "say why in their tooltip" goal is untestable. Drop it or move it to a toolbar button.

ZO-6 MAJOR. Cross-draft. OC-04 (session tabs) and ZP-21 (workspace tabs) also record transitions and bind chords. State whether history is per tab or per window. INDEX calls OC-05 "small, self-contained", which ZO-1 and ZO-2 contradict.

ZO-7 NIT. Lines 4, 27. The RootView link uses a repo-relative path from a docs folder, so it does not resolve.

ZO-8 NIT. Lines 92-93, 117-120. "O(1) push and trim" at 100 entries is noise; the mutation-check sentence and the 21 languages / ten gates boilerplate belong in INDEX or the test plan.

ZO-9 NIT. Line 42. The 100-entry cap is copied from upstream, not derived from a need. Say so.

Reuse check: no existing back-stack code exists; a small `NavigationHistory` model is reasonable.

## Inventive Catalyst reviewer (IC)

No Blockers. Two real contradictions are accepted without examination.

IC-1 MAJOR. Lines 26-30, 72-88, file 2. The doc treats "record every transition" and "walk the stack without mutating it" as compatible. With an onChange recorder, going back pushes a new entry and wipes the forward stack, so Cmd+] breaks. Fix (inversion plus intermediary): put history next to the model owning `activeSection`, `selectedChat` and the settings tab, route all through one `navigate(to:)` entry, and have back/forward use a `restore(_:)` path with a replaying flag or a no-op when the target equals the cursor entry. State cursor semantics, including truncating forward entries on a fresh push. Liability: every existing direct assignment must be found; a missed one silently skips recording; the diff is wider than "small".

IC-2 MAJOR. Lines 34-35, 46-51, CUJ 2. Entries hold chat ids but chats can be deleted, archived or moved during a session, and sections like `.server` are conditional. Failure is undefined and the menu item may show enabled. Fix: validate on back/forward through an injected resolver, skip invalid entries, prune dead ids, derive can-go-back rather than caching it, and test the skip.

IC-3 NIT. Lines 40-41, 55-61. The collapse rule does not say which pane the surviving entry stores. State: a settings push replaces the top entry if the top is also settings; going back into a settings entry and choosing another pane replaces in place.

IC-4 NIT. Lines 43-44, 96-97. Menu tooltips do not display; "menu items announce enabled state" is system behavior, not a design requirement.

IC-5 NIT. Lines 38-39, 83. Cmd+[ and Cmd+] are indent and outdent in text fields and the composer; collision with the composer and OC-01 is unexamined. The `AlternateShortcutBridge` box adds nothing.

IC-6 NIT. INDEX sequencing. OC-04 and ZP-21 will add tab identity. Keep the entry a small struct or enum so it can be extended. Do not add it now.

Passed: 100-entry bound with array and cursor; no persistence and no history UI; reuse of the Commands and catalog pattern.
