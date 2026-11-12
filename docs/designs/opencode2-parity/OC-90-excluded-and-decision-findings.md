# OC-90: Excluded and decision findings

The eleven findings from [OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md)
that did not become drafts: excluded under the offline/local-first
policy, N/A on this platform, covered elsewhere, or consciously
deferred. The policy baseline is the roadmap exclusion set already
approved for the ZCode program
([ZP-90](../zcode-parity/ZP-90-excluded-and-decision-findings.md)): no
cloud services or accounts, no remote workspaces, no non-macOS rows.

* Author: whit3rabbit
* Status: Draft (decision document)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Summary table

| Finding | Verdict | Decision |
|---|---|---|
| 1 multi-window N windows | excluded (single-window model) | do not adopt; tabs instead |
| 1 crash-reporter/telemetry half | excluded (no telemetry posture) | do not adopt |
| 3 provider OAuth + accounts | excluded permanently | never cloud OAuth; API keys per ZP-22 |
| 4 SSH remote servers | excluded (remote category) | do not adopt |
| 4 WSL | N/A | record only, macOS-only app |
| 4 pairing (QR + remote access) | excluded (remote control axis) | do not adopt |
| 5 cloud usage and budget dialog | excluded (no cloud tier) | do not adopt |
| 6 motion tuning system | excluded (YAGNI, recorded) | do not adopt; reopenable |
| 6 translation governance | docs task, not a design | implement as a docs change |
| 7 process-daemon (PTY persistence) | deferred (conflicts with terminal spec) | revisit after inspector-terminal |
| 7 HTTP recording cassettes | excluded (dev test tooling) | do not adopt |

## Per-finding analysis

### Section 1: multi-window model (single-instance half is covered)

OpenCode runs N peer main windows with per-window state. TurboSpark is
deliberately single-window; the adopted answers for the same underlying
needs are [ZP-21](../zcode-parity/ZP-21-workspace-tabs.md) workspace
tabs, [OC-04](OC-04-session-tabs.md) session tabs, and ZP-23
concurrent chats. The finding's other halves are already owned: the
single-instance lock and second-launch focus live in the
[desktop-shell-hardening](../../../.kiro/specs/desktop-shell-hardening/brief.md)
spec, and the background lifecycle half is the app's existing strength
(MenuBarExtra dashboard plus keep-running-in-background setting).
* **Decision**: do not adopt multi-window. Reopen only if a real
  multi-display workflow cannot be served by tabs and split panes.

### Section 1: renderer crash recovery, telemetry half

The Electron-side half of this finding (crash reporter, net-log,
crashpad) has no honest local analog short of adding a crash reporting
service, which is telemetry. The platform-moot half (renderer process
loss) does not exist on AppKit. The portable residuals (hang detection,
guarded relaunch, diagnostics export) are drafted as
[OC-14](OC-14-hang-watchdog.md) and owned by the shell-hardening spec
respectively.
* **Decision**: do not adopt crash reporting. OC-14 logs locally and
  exports via the spec's redacted bundle.

### Section 3: provider connect, accounts, and OAuth

OpenCode connects cloud providers with OAuth device flows and multiple
named accounts. TurboSpark's permanent decision (author, 2026-11-11,
recorded in ZP-90 for the ZCode program) applies with equal force
here: the app never adds cloud OAuth, hosted identity, or account
login. Remote inference via user-managed API keys is already adopted
as [ZP-22](../zcode-parity/ZP-22-remote-providers.md) for the ZCode
program; that draft covers this finding's portable half for OpenCode's
provider page as well, whenever the app wants it.
* **Decision**: excluded permanently (OAuth); API-key path is ZP-22,
  not re-drafted here.

### Section 4: SSH remote server connections

The roadmap excludes remote workspaces. SSH would move the agent
execution trust boundary off-device, which the permission ladder and
sandbox designs do not model. Same reasoning as ZP-90's 6.5.
* **Decision**: do not adopt.

### Section 4: WSL

Windows-only by construction; the report itself records it as
legitimately N/A.
* **Decision**: record only.

### Section 4: pairing (QR pairing with mobile/remote)

A phone companion driving desktop sessions needs a pairing protocol
and a remote-control surface into a loopback-only app. This is the
same axis ZP-90 excluded for the ZCode program (its findings 1.4 and
6.6): a LAN-only adaptation is feasible engineering but creates a
remote control surface that deserves its own security design, not a
parity port. ZP-18 (mobile *device control*, simulator and emulator
agent tools) is a different feature and stays adopted there.
* **Decision**: do not adopt. Reopen only with a dedicated remote
  access security design.

### Section 5: cloud usage and budget dialog

OpenCode shows signed-in workspace allowances and Console budgets.
TurboSpark has no account, workspace, quota, or credit concept, and
the report itself says nothing should be ported unless a cloud tier is
ever added. The honest local analog (token ledger, usage dashboard)
already exists.
* **Decision**: do not adopt.

### Section 6: motion tuning system

OpenCode's tuning subsystem lets designers tune spring parameters live
and ship winning values, across a large motion library. TurboSpark's
motion vocabulary is a handful of named constants in
[TurboSparkMotion.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Theme/TurboSparkMotion.swift)
governed by the design-check ratchet and the design system. A tuning
UI would outgrow its subject: there are six-ish knobs, and the
existing design-token pipeline is the tuning loop.
* **Decision**: do not adopt. Reopen if the motion vocabulary grows
  enough that live tuning beats editing constants (roughly: more than
  a dozen animated sites sharing tunable parameters).

### Section 6: translation governance rules

OpenCode's desktop AGENTS.md encodes a 31-line translation policy:
corpora consultation, official dictionaries, terminology verification,
and recorded provenance. This is governance prose, not a feature; a
design draft is the wrong artifact.
* **Decision**: implement as a docs change, not a design. Concretely:
  add a "Translation sources" subsection to
  [swift/docs/SWIFT_LOCALIZATION.md](../../../swift/docs/SWIFT_LOCALIZATION.md)
  naming the authorities consulted for the 21 languages (Unicode CLDR
  terminology plus the Apple/Microsoft style guides per language) and
  require new locales to record their sources. This is recommended,
  small, and needs no OpenCode code reading beyond the policy
  reference.

### Section 7: process-daemon (opencode-pty PTY service)

OpenCode's standalone PTY service keeps terminal state alive across
app and server restarts with handoff tickets. The
[inspector-terminal](../../../.kiro/specs/inspector-terminal/brief.md)
spec deliberately terminates every PTY process group on app quit with
no orphans; a surviving daemon contradicts that contract and adds a
token-authenticated socket service to a loopback-only app. The
terminal spec's session store already solves the in-app problem (a
hidden pane keeps its shell alive).
* **Decision**: deferred, not drafted. Reopen only if users ask for
  shells surviving app restarts. Any such work is an amendment to the
  terminal spec with its own trust-boundary review, not a parity
  draft.

### Section 7: HTTP recording (cassettes)

OpenCode's recorder is a published npm test utility, not an app
feature; the report says as much. The Swift app has no provider HTTP
traffic to record. Generation is in-process FFI. MCP, hub downloads,
and web tools already test against stubs and fixtures; the Rust side
owns provider fixtures by convention.
* **Decision**: do not adopt. If deterministic HTTP tests are ever
  needed in Swift, the OpenCode package is the reference design to
  consult.

## Decision log

* 2026-11-11: OpenCode2 program classified; offline/local-first
  policy confirmed as the exclusion baseline; this record written.
* Reopen conditions are recorded per finding above; none of these
  exclusions is permanent except provider OAuth (permanent by author
  decision 2026-11-11, inherited from ZP-90).

--------------------------------------------------------------------------------

*Drafted with EGM*
