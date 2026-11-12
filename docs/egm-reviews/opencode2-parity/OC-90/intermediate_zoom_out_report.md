# Stage 8: Intermediate Zoom Out report (OC-90)

Reviewer: egm-zoom-out (completed), given the Stage 5, 6 and 7 reports. Verdict: the document is at the right altitude and has enough context. No Blocker. YAGNI, premature optimization and code reuse barely apply (no new code is proposed); they bite only on the translation docs task and the reopen triggers.

Major (IDs S8-M1 to S8-M8)

* S8-M1, 8-9, 164-168: baseline "no cloud services or accounts" is contradicted by adopted ZP-22 (hosted APIs with user keys). ZP-90 draws a narrower line: no cloud OAuth, hosted identity or app-operated infrastructure. State the narrow rule once in the preamble.
* S8-M2, 166-168: the closing claims are false or unclassified. Reopen conditions exist only for multi-window (46), pairing (96), motion tuning (117), PTY daemon (146), and weakly cassettes (158). "None permanent except OAuth" conflicts with WSL (N/A). Add a class or reopen column to the table (permanent by decision, N/A, policy exclusion reopenable on policy change, docs task) and rewrite the log from it, as ZP-90 163-170 does.
* S8-M3, 21, 45-47: the multi-window reopen condition is true by definition because tabs and split panes never serve a second display. Restate it. Row should read "excluded, partly covered" to match INDEX.
* S8-M4, 99-106, 151-160: cloud usage and cassette reasons are stale after ZP-22 (third-party cost; Swift SSE clients needing fixtures). Decisions can stand; rewrite reasons and route cost display and fixtures to ZP-22 follow-ups (ZP-90 line 131).
* S8-M5, 49-59: crash reporting equated with telemetry. Narrow to "no transmission"; local capture is covered by OC-14 and the shell-hardening export.
* S8-M6, 136-149: PTY daemon framed as binary; respawn with restored cwd, env and scrollback is an alternative. Table (30, "after inspector-terminal") and body (146, "if users ask") give different triggers.
* S8-M7, 87-97, 143: "loopback-only" ignores the explicit loopback-or-single-verified-Tailscale-IPv4 bind rule in ZP-90 (crates/server/src/bind.rs exists). Cite it in the reopen note.
* S8-M8, 121-134: translation governance is the one non-exclusion, with no owner, acceptance criteria or tracking. Requirement is buried; "needs no OpenCode code reading" is unsupported; CLDR is locale data, not a terminology authority (131).

Nit

* S8-N1, 113: broken TurboSparkMotion.swift link.
* S8-N2, 114-119: "six-ish knobs" inexact (about nine named animations); trigger counts a different quantity.
* S8-N3, 3: "eleven" counting rule not given; INDEX arithmetic (16 + 8 + 11) does not reconcile to 31.
* S8-N4, headings and table: "Section N" and "Finding N" refer to the external report; add a one-line topic per table row.
* S8-N5, 40, 94, 78, 93: ZP-23, ZP-18 and ZP-90 section numbers are bare references.
* S8-N6, 64-65, 167-168: OAuth permanence has no link to ZP-90 section 7.2.
* S8-N7, 52-56: "respectively" pairs three residuals with two owners.
* S8-N8, 61-72: multiple named accounts is not OAuth; ZP-22 coverage of it is unbacked.
* S8-N9, 101-105: "analog already exists" overstates (source rates Partial).
* S8-N10, 156-157: "Rust owns provider fixtures by convention" uncited.
* S8-N11, table: verdict labels inconsistent.
* S8-N12, 172: footer matches INDEX, keep.

Not verified: .kiro specs (desktop-shell-hardening, inspector-terminal).
