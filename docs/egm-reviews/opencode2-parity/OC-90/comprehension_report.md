# Stage 6: Comprehension report (OC-90)

Reviewer: egm-comprehension (completed). Read the document alone, no context files. No Blocker.

## What this proposes

The document is a decision log for eleven findings from a comparison report (OPENCODE2_COMPARISON.md) between the OpenCode desktop app and TurboSpark. None of the eleven became design drafts. Most are rejected because TurboSpark is offline, local-first and macOS-only: no cloud accounts or OAuth, no remote workspaces (SSH, phone pairing), no telemetry, no cloud usage dialogs, no Windows (WSL). Two items are special cases. Translation governance becomes a docs change in SWIFT_LOCALIZATION.md. The PTY daemon is deferred because it conflicts with the terminal spec's "no orphan processes" rule. Each entry gives a reopen condition, except provider OAuth, which is permanently excluded.

## How the existing system works, per this document

TurboSpark is a single-window macOS app with a Swift UI that calls a Rust core through in-process FFI. It runs loopback-only, with no network-facing services. Multiple concerns are handled by tabs and split panes, and a MenuBarExtra dashboard with a keep-running-in-background setting covers the background lifecycle. A "permission ladder" and sandbox designs bound agent execution, and the terminal spec kills every PTY process group on quit. Motion is a handful of named constants governed by a "design-check ratchet". Remote inference uses user-managed API keys (ZP-22). The document does not explain several of these mechanisms; it only names them.

## Findings

* CO-1 Major, 3-4, 19-31, section headings: findings are identified only by "Finding N" and "Section N" numbers from the external report; the table has no one-line topic per finding. "The OpenCode2 program" (164, 166) is undefined.
* CO-2 Major, 7-9, 77-78, 91-92, 64: ZP-90 references are bare section numbers ("6.5", "1.4 and 6.6"); the SSH and pairing reasoning is delegated to them and cannot be checked from this file.
* CO-3 Major, 94, 40: ZP-18 and ZP-23 are cited without links or definitions. "ZP-" and "OC-" prefixes and the "ZCode program" are never explained (also Nit, 7, 64).
* CO-4 Major, 78-79: "permission ladder and sandbox designs" are named but not linked or defined, yet carry the SSH rejection.
* CO-5 Major, 42-45, 138-141: desktop-shell-hardening and inspector-terminal specs carry load-bearing claims with no summary of what they require.
* CO-6 Major, 52-56: crash-recovery paragraph is hard to follow. "Platform-moot half" is unexplained. "Respectively" pairs three residuals (hang detection, guarded relaunch, diagnostics export) with two owners without a clear mapping.
* CO-7 Major, 3, 17-31: "eleven findings" counts sub-items (rows split from findings 1, 4, 6, 7) but the counting rule is not stated.
* CO-8 Nit, 21 vs 35: table says "tabs instead", body says "tabs and split panes"; split panes are not otherwise explained.
* CO-9 Nit, 112-119: motion link does not resolve from this folder. "Design-check ratchet" is undefined. "Six-ish knobs" and "roughly more than a dozen" are vague.
* CO-10 Nit, 127-134: "21 languages" not listed. "Needs no OpenCode code reading beyond the policy reference" refers to an unmentioned step. AGENTS.md not explained.
* CO-11 Nit, 66-70: "whenever the app wants it" gives no adoption trigger for ZP-22.
* CO-12 Nit, 158-160: "the OpenCode package" is not named. "Cassettes" is not defined.
* CO-13 Nit, 166-168: "reopen conditions are recorded per finding above" is only partly true (crash reporting, SSH, cloud usage, WSL lack one).
* CO-14 Nit, 60-62 vs 12: ZP-22 is called "already adopted" while this doc is Draft; unclear if ZP-22 is implemented or only designed.
