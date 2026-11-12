# Stage 6 report: comprehension (INDEX.md, read alone)

## What this proposes

This is the index for a program of design drafts, OC-01 to OC-14 plus OC-90. The drafts aim to account for all 31 "Missing in Swift" findings from an audit comparing the OpenCode desktop app with this native-Swift, macOS-only, local-first inference app. Fourteen drafts cover 16 findings. Eight more findings are owned by other specs or by the ZCode-parity drafts (ZP-xx). OC-90 records eleven findings as excluded or decided. The document also lists cross-draft dependencies, an eight-step build order, shared constraints (offline, localization, permissions) and where the findings came from.

## How the existing system works, per this document

The existing app is a native Swift macOS app with a Rust component. The document names one Rust crate for ACP and one "engine" for the version handshake. It has a settings.json, a Keychain, a settings-control catalog, a tool-permission "ladder" (AppToolPermissionEngine), a localization pipeline (ten gates, 21 languages) and a make design-check step. It has an unread/pending-badge feature (ZP-02) and a concurrent-chats feature (ZP-23), which the document treats as sequencing prerequisites. It also has local audio models and an edit tool that can host post-edit hooks. The document never says whether the ZP drafts and the desktop-shell-hardening and inspector-terminal specs are built, drafted or only planned. A fresh reader cannot tell which of these exist today.

## Findings (CO)

CO-1 Major. Lines 3-5, 28-69. Counts do not reconcile: 16 + 8 + 11 = 35, not 31. Multi-window appears in the covered table (53) and in OC-90 (64); session tabs and the unread half also appear twice. Fix: one row per finding, one disposition each, summing to 31.

CO-2 Major. Lines 33-47, 51-60. Finding numbers (1, 3, 4, 7, 8, 9a) are undefined. "3" labels keybinds, notifications and the API-key path; "9a" is unexplained. Fix: define them as section numbers of OPENCODE2_COMPARISON.md.

CO-3 Major. Lines 53 and 64. Multi-window is "covered" by the shell-hardening spec in one table and "excluded" in OC-90. Fix: state one disposition, or say which half each owns.

CO-4 Major. Lines 95, 99. Step 3 puts OC-11 after OC-10's seam, but OC-10 is step 7 and OC-11 is listed again there. Fix: list OC-11 in step 7 only.

CO-5 Major. Lines 94, 76-78, 58. OC-03 is "app-local" in step 2 but needs ZP-23 first (and ZP-02). Fix: annotate the prerequisites or move it.

CO-6 Major. Lines 56-60, 71-78. Many items depend on ZP-02/21/22/23 with no status or order given. Fix: a status column for each ZP prerequisite and a slip rule.

CO-7 Major. Lines 85-87. OC-12/OC-13 dependency is vague ("once a daemon-based transport ever exists"); "the engine" is undefined; OC-13 is step 1 with no stated consumer. Fix: define engine, name the consumer or say none today.

CO-8 Major. Lines 102-114. "No Python or Node runtime in the product path" may conflict with OC-10 (LSP servers are often Node or Python). Fix: one line on which language servers OC-10 supports and how it complies.

CO-9 Nit. Lines 1-12, 21, 108-114. Undefined terms: EGM, ZCode, OpenCode, "Missing in Swift", "dock badge spec", "Working rows", SettingsControlCatalog, permission ladder, "ten localization gates", "local green is not release evidence". Fix: link or define at first use.

CO-10 Nit. Lines 117-120. "the user's own LAN-facing choices (none drafted)" is confusing. Fix: "no LAN features are drafted".

CO-11 Nit. Lines 126-131. Commit 7ce35879b8 and branch feat/classlru-replay-arm lack labels. Fix: "OpenCode commit" and "Swift-side audited branch".

CO-12 Nit. Lines 34, 64. OC-14 is the "residual of the crash-recovery finding" but that finding is never named; unclear whether the crash-reporter half is one finding or two. Fix: name it and say which half each owns.

CO-13 Nit. Line 30. "14 drafts covering 16 findings" is checkable only if the reader knows OC-07 and OC-10 each cover two. Fix: say so in the heading.
