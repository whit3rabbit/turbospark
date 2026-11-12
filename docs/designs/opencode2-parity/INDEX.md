# OpenCode2 parity draft program

Design drafts for every "Missing in Swift" finding in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md) (audited
2026-10-31, OpenCode desktop `7ce35879b8`). Goal statement for the
program: account for all 31 Missing findings and implement the ones
that fit a local-first, macOS-only, native-Swift inference app.

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## What this folder is

One design draft per missing feature, following the EGM design
document template. Each draft cites its finding, the OpenCode sources,
the Swift sources, and an implementation plan grounded in this
repository. Decisions with more than one defensible answer are
recorded as Open Questions, not silently chosen.

Companion program: the ZCode comparison has its own draft folder,
[docs/designs/zcode-parity/](../zcode-parity/INDEX.md). Several
findings describe the same feature; those cross-reference it instead
of duplicating.

## Coverage map (all 31 findings)

### Drafted in this folder (14 drafts covering 16 findings)

| Finding | Draft | Size note |
|---|---|---|
| 1 hang detection and guarded relaunch (residual of the crash-recovery finding) | [OC-14](OC-14-hang-watchdog.md) | small-medium; crash-reporter half excluded in OC-90 |
| 3 keybind customization | [OC-01](OC-01-keybind-customization.md) | medium |
| 3 notifications and sounds | [OC-02](OC-02-notifications-and-sounds.md) | medium; composes ZP-02 and the dock badge spec |
| 4 sources panel | [OC-09](OC-09-sources-panel.md) | medium |
| 7 voice call mode (local) | [OC-08](OC-08-local-voice-call.md) | large; pure integration of the existing local audio stack |
| 7 first-run onboarding + splash | [OC-07](OC-07-onboarding-and-splash.md) | medium; one system, two findings |
| 8 LSP diagnostics + symbol tools | [OC-10](OC-10-lsp-diagnostics-symbols.md) | large; one system, two findings |
| 8 formatter pass | [OC-11](OC-11-formatter-pass.md) | small-medium |
| 8 ACP adapter | [OC-12](OC-12-acp-adapter.md) | large; new Rust crate |
| 8 version handshake | [OC-13](OC-13-engine-version-handshake.md) | small |
| 9a tab strip, draft tabs, closed-tab restore | [OC-04](OC-04-session-tabs.md) | medium-large; composes ZP-21 and ZP-23 |
| 9a inbox triage mode | [OC-03](OC-03-inbox-triage-mode.md) | small-medium; composes ZP-02 and ZP-23 |
| 9a back/forward history | [OC-05](OC-05-back-forward-history.md) | small |
| 9a sidebar hover peek | [OC-06](OC-06-sidebar-hover-peek.md) | small |

### Already covered elsewhere (no new draft)

| Finding | Owned by |
|---|---|
| 1 multi-window: single-instance + second-launch focus + deep links | [desktop-shell-hardening](../../../.kiro/specs/desktop-shell-hardening/brief.md) spec |
| 1 diagnostics export | desktop-shell-hardening spec (rotating log, redacted zip, support report) |
| 4 terminal UI pane | [inspector-terminal](../../../.kiro/specs/inspector-terminal/brief.md) spec (SwiftTerm, PTY, lifecycle) |
| 4 plugin runtime | [ZP-05](../zcode-parity/ZP-05-builtin-marketplace-seeding.md) (seeding) + [ZP-06](../zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md) (page host); JS bundle evaluation excluded |
| 9a global command palette | [ZP-16](../zcode-parity/ZP-16-command-center-palette.md); add a settings-row scope reusing `SettingsControlCatalog` when built |
| 9a unread/pending state half | [ZP-02](../zcode-parity/ZP-02-unread-pending-badges.md); OC-02 covers the notification channel, OC-03 the inbox view |
| 3 provider API-key path | [ZP-22](../zcode-parity/ZP-22-remote-providers.md) (adopted 2026-11-11); OAuth excluded permanently |
| 9a session tab strip (workspace layer) | [ZP-21](../zcode-parity/ZP-21-workspace-tabs.md); OC-04 adds the session layer |

### Excluded or decided

[OC-90](OC-90-excluded-and-decision-findings.md) records the eleven
non-drafted findings: multi-window, crash-reporter telemetry, provider
OAuth (permanent), SSH, WSL, QR pairing, cloud usage dialog, motion
tuning (YAGNI, reopenable), translation governance (docs task with a
concrete recommendation), the PTY persistence daemon (deferred), and
HTTP recording.

## Cross-draft dependencies

* OC-02 (notifications) consumes the same state as ZP-02's badges;
  sequence ZP-02 before OC-02 goes live so the two share one state
  source.
* OC-03 (inbox) and OC-04 (session tabs) both compose on ZP-23
  (concurrent chats) for live Working rows and multi-open sessions.
  Sequence ZP-23 first.
* OC-04's chords join [OC-01](OC-01-keybind-customization.md)'s
  override store; build OC-01 before exposing tab chords.
* OC-07's import step hands off to the desktop-shell-hardening import
  wizard; until that spec lands, the step is absent, not a placeholder.
* OC-10 (LSP) and OC-11 (formatter) both hook after the edit tool;
  land OC-10's post-edit seam first and have OC-11 reuse it.
* OC-12 (ACP) stands alone (Rust crate), but its docs must reference
  the version handshake (OC-13) once a daemon-based transport ever
  exists.

## Suggested sequencing

Small, independent wins first, then the cross-cutting systems:

1. OC-13, OC-05, OC-06 (small, self-contained)
2. OC-01, OC-09, OC-03 (medium, app-local)
3. OC-02, OC-14 (medium; OC-02 after ZP-02)
4. OC-07 (onboarding; coordinate the import step with the spec)
5. OC-04 (session tabs; after ZP-21 and ZP-23)
6. OC-08 (voice call; large integration)
7. OC-10, OC-11 (LSP and formatter pair)
8. OC-12 (ACP; largest surface, standalone crate)

## Program constraints

The same constraints as the [zcode-parity
program](../zcode-parity/INDEX.md): macOS only; Rust edition 2021,
MSRV 1.82; no Python or Node runtime dependency in the product path;
every UI string through bundled `Text` or `AppStrings.localized`
passing the ten localization gates in all 21 languages in the same
change; new surfaces follow [DESIGN.md](../../../DESIGN.md) and
`make design-check`; secrets stay out of settings.json; Keychain use
opt-in per credential; permissions run through the
`AppToolPermissionEngine` ladder and any agent-facing tool draft states
its ladder position; local green is not release evidence
([verification](../../../.claude/docs/verification.md)).

**Offline constraint (user directive 2026-11-11)**: this program adds
no login, no account, no OAuth, no cloud storage, and no hosted
service of any kind. Every drafted feature runs fully on device or on
the user's own LAN-facing choices (none drafted). Cloud-shaped
findings are excluded in [OC-90](OC-90-excluded-and-decision-findings.md)
rather than adapted.

## Provenance

Findings and links come from
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md), audited
2026-10-31 at upstream commit `7ce35879b8` against branch
`feat/classlru-replay-arm`. The report notes its Swift absence claims
are grep-and-read evidence; drafts repeat those citations without
re-verifying every one. Per repo rules, re-check a link before acting
on a single finding. Drafts were written 2026-11-11.

*Drafted with EGM*
