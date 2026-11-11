# ZCode parity draft program

Design drafts for every "Missing in Swift" finding in
[ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md) (audited 2026-10-31, ZCode
`a60ba6a7` / v3.17.1). Goal statement for the program: account for all 40
Missing findings and implement the ones that fit a local-first, macOS-only,
native-Swift inference app.

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## What this folder is

One design draft per missing feature, following the EGM design document
template. Each draft cites its finding number, the ZCode sources, the Swift
sources, and an implementation plan grounded in this repository. Decisions
that have more than one defensible answer are recorded as Open Questions in
the draft rather than silently chosen.

## Coverage map (all 40 findings)

### Drafted in this folder

| Finding | Draft | Size note |
|---|---|---|
| 1.2 off-peak idle tasks | [ZP-01](ZP-01-off-peak-idle-tasks.md) | medium |
| 2.3 unread + pending-interaction row badges | [ZP-02](ZP-02-unread-pending-badges.md) | medium, notes dependency on concurrent chats (finding 2.1) |
| 2.14 model trajectory viewer | [ZP-03](ZP-03-model-trajectory-viewer.md) | large, adds Rust-side rollout logging |
| 3.5 subagent-to-coordinator messaging | [ZP-04](ZP-04-subagent-coordinator-messaging.md) | medium |
| 4.2 builtin marketplace seeding + CDN zip | [ZP-05](ZP-05-builtin-marketplace-seeding.md) | medium |
| 4.3 + 4.4 + 4.5 UI plugins, page sandbox, Gen UI | [ZP-06](ZP-06-ui-plugins-sandbox-gen-ui.md) | largest draft; one system, three findings |
| 4.10 plugin creator | [ZP-07](ZP-07-plugin-creator.md) | small |
| 5.7 office document rendering | [ZP-08](ZP-08-office-document-rendering.md) | large |
| 5.8 whiteboard canvas | [ZP-10](ZP-10-whiteboard-canvas.md) | medium |
| 5.9 image preview annotation | [ZP-11](ZP-11-image-annotation.md) | medium |
| 5.11 treemapping activity | [ZP-12](ZP-12-treemap-activity.md) | small |
| 1.16 dev-build icon badge | [ZP-13](ZP-13-dev-build-icon-badge.md) | small |
| 6.9 browser page permission governance | [ZP-14](ZP-14-browser-page-permissions.md) | medium |
| 6.12 browser WebM recording | [ZP-15](ZP-15-browser-webm-recording.md) | medium; reverses an earlier exclusion, see draft |
| 8.5 command center palette | [ZP-16](ZP-16-command-center-palette.md) | medium; supersedes an earlier deferral, see draft |
| 8.6 error boundaries | [ZP-17](ZP-17-error-boundaries.md) | medium; SwiftUI has no crash-isolation primitive |
| 9.1 office document generation | [ZP-09](ZP-09-office-document-generation.md) | large; must respect the no-Python/Node rule |
| 9.5 mobile device control | [ZP-18](ZP-18-mobile-device-control.md) | medium |
| 9.10 web image search | [ZP-19](ZP-19-web-image-search.md) | small |
| 9.14 utility plugins (portable half) | [ZP-20](ZP-20-bundled-content-plugins.md) | small-medium |
| 1.6 workspace tabs | [ZP-21](ZP-21-workspace-tabs.md) | large; composes on the adopted concurrent chats (ZP-23) |
| 7.1 (+7.2 key half) remote providers | [ZP-22](ZP-22-remote-providers.md) | large; adopted 2026-11-11, API keys only, never OAuth |
| 2.1 concurrent background chats | [ZP-23](ZP-23-concurrent-chats.md) | large; adopted 2026-11-11 (finding was Different) |

### Already owned by an existing kiro spec (do not re-draft)

| Finding | Existing spec |
|---|---|
| 1.5 single-instance lock + deep links | [.kiro/specs/desktop-shell-hardening](../../../.kiro/specs/desktop-shell-hardening/brief.md) |
| 5.1 git staging, commit, push | [.kiro/specs/git-write-integration](../../../.kiro/specs/git-write-integration/brief.md) |
| 5.5 per-turn change review + safe file rewind | [.kiro/specs/git-write-integration](../../../.kiro/specs/git-write-integration/brief.md) |
| 5.3 workspace file tree sidebar | [.kiro/specs/inspector-files](../../../.kiro/specs/inspector-files/brief.md) |
| 6.4 terminal emulator UI | [.kiro/specs/inspector-terminal](../../../.kiro/specs/inspector-terminal/brief.md) |

### Excluded or decided (one consolidated record, not fabricated designs)

[ZP-90](ZP-90-excluded-and-decision-findings.md) covers the remaining 12
findings: 1.4, 1.10, 3.6, 6.5, 6.6, 7.1, 7.2, 7.3, 7.4, 7.9, 7.11, 7.12.
For each it records what ZCode has, what a local-first adaptation would
require, and the decision. Nine stay excluded under the policy already
approved in [.kiro/steering/roadmap.md](../../../.kiro/steering/roadmap.md)
(cloud services and accounts, remote workspaces, non-macOS rows); 7.9's
portable half (log bundle export) is inside the `desktop-shell-hardening`
spec; 7.1 is adopted as ZP-22 (upstream APIs, API keys only); 7.2 is
excluded permanently by author decision 2026-11-11: the app never adds
cloud OAuth.

## Cross-draft dependencies

* ZP-23 (concurrent chats, adopted 2026-11-11) is the prerequisite for
  ZP-02's live badge states and ZP-21's tabs holding generating chats.
  Sequence ZP-23 before both.
* ZP-09 (office generation) needs ZP-08 (office rendering) for review of
  what the agent wrote, and ZP-19 (image search) fills the role ZCode's
  office plugins delegate to `image_search`.
* ZP-06 (UI plugins / sandbox / Gen UI) is the host for what ZP-07's
  created plugins could declare, and ZP-20's guide skills document the rest
  of the plugin system.
* ZP-05 (builtin seeding) is the delivery mechanism ZP-20 (bundled content
  plugins) and ZP-18 (device control plugins) would ship through.
* ZP-22 (remote providers) stands alone but touches the turn loop seam;
  its first task verifies that seam before anything else lands.

## Suggested sequencing

Small, independent wins first, then the systems with Rust-side or
cross-cutting work:

1. ZP-13, ZP-12, ZP-19, ZP-07 (small, self-contained)
2. ZP-01, ZP-11, ZP-10, ZP-14 (medium, app-local)
3. ZP-16, ZP-17, ZP-20, ZP-18, ZP-15 (medium, some cross-cutting)
4. ZP-05, ZP-04, ZP-03 (add persistence or protocol machinery)
5. ZP-08, ZP-09 (office pair)
6. ZP-23 (concurrent chats: the big behavioral change; land before ZP-02
   goes live and ZP-21 builds on it)
7. ZP-02, ZP-21 (compose on ZP-23)
8. ZP-22 (remote providers; verify the turn-loop seam first)

## Program constraints (from the root AGENTS.md and the roadmap)

* macOS only; runtime path is Metal via `crates/gpu`; Rust edition 2021,
  MSRV 1.82.
* No Python or Node runtime dependency in the product path. ZCode features
  that lean on docx-js, python-pptx, or xterm.js need native replacements
  or skill-level delegation to user-installed tooling, and the drafts say
  which.
* Every UI string uses bundled `Text` or `AppStrings.localized` and passes
  the ten localization parity gates; new surfaces follow
  [DESIGN.md](../../../DESIGN.md) and the design check
  (`make design-check`).
* Secrets stay out of settings.json; Keychain use is opt-in per credential
  and never at launch for an unused feature.
* Permissions run through the `AppToolPermissionEngine` ladder. A draft
  adding an agent-facing tool must state its ladder position.
* Local green is not release evidence. Verification gates apply as documented in
  [.claude/docs/verification.md](../../../.claude/docs/verification.md).

## Provenance

Every draft repeats its finding's ZCode and Swift links from
ZCODE_COMPARISON.md. Author decisions recorded 2026-11-11: ZP-22
(remote providers) and ZP-23 (concurrent chats) adopted; 7.2 cloud OAuth
excluded permanently; ZP-15 and ZP-16 reverse their earlier kiro
deferrals. The report was link-verified at audit time; the linked
files can move after it. Per repo rules, re-check a link before acting on a
single finding. Drafts were written 2026-11-11 against branch
`feat/classlru-replay-arm`.

*Drafted with EGM*
