# ZP-20: Bundled content plugins (portable half of finding 9.14)

Finding 9.14 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md), portable half.
Parity targets: ZCode's
[zcode-guide plugin](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/zcode-guide-plugin/package.json)
and the visualize skill's
[output contract](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/visualize-plugin/skills/visualize/SKILL.md)
(related finding 9.9, Partial).

* Author: whit3rabbit
* Status: Implemented
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Ship a self-diagnosis guide skill pack (the portable core of ZCode's
zcode-guide) and a visualize-style output contract skill (the portable
core of finding 9.9) as bundled content. The two inapplicable thirds of
finding 9.14 (site publishing and legacy session restore) are recorded
as excluded in [ZP-90](ZP-90-excluded-and-decision-findings.md).

## The Problem

TurboSpark ships zero built-in content skills; its built-in content is
four agent definitions. When a user's hooks, commands, skills, MCP
servers, or plugins misbehave, the agent has no guided diagnosis path,
and the user has no in-app documentation surface for the agent to cite.
Separately, when the agent should explain data visually, nothing tells
it the output contract that makes interactive HTML render reliably in
the artifact panel.

## Goals

* A `turbospark-guide` bundled skill pack: a configuration guide skill
  plus five diagnosis skills (hooks, commands, skills, mcp, plugins),
  each teaching the agent to inspect the relevant stores, list state,
  and walk the user through verification, referencing the app's actual
  paths and settings panes.
* A `visualize` bundled skill: a strict output contract for agent-written
  interactive HTML (fragment-free single file, size cap 1 MB, no
  external CDN by default, self-contained charts) targeting the existing
  `ArtifactWebView` surface, with guidance for common chart forms.
* Both ship through the bundled-skill root (ZP-07's mechanism) and are
  unremovable but disable-able like other skills.

## Non-Goals

* Not the ZP-06 plugin UI layer; visualize output renders through the
  existing artifact panel and its existing grants.
* No bundled charting library; the skill instructs self-contained SVG
  or canvas code (option (b) from ZP-06's open question, applied here).
* Site publishing and legacy session restore: excluded (ZP-90).

## Open Questions

* **Guide content source of truth**: hand-written skill text drifts from
  the app. Options: (a) hand-written with a docs cross-check note;
  (b) generated from a paths manifest. Lean (a) with a maintenance note;
  author decides.

## Critical User Journeys

### CUJ 1: diagnose MCP

* **Given**: an MCP server fails to spawn.
* **When**: the user asks "why is my MCP server not working".
* **Then**: the agent activates the mcp diagnosis skill, inspects the
  server config, spawn env, and log output, and reports the failing
  step with the exact settings pane to fix it.

### CUJ 2: visualize a dataset

* **Given**: the user asks to visualize benchmark numbers.
* **When**: the agent activates the visualize skill.
* **Then**: it writes a self-contained HTML file within the contract
  (size cap, no CDN), the artifact panel renders it interactively, and
  a "save as" gives the user the file.

### CUJ 3: guide citation

* **Given**: the user asks how shortcuts or profiles work.
* **When**: the agent activates the configuration guide skill.
* **Then**: the answer cites the app's real panes and paths instead of
  generic advice.

## Technical Plan (Architecture)

```text
Bundled-skill root (ZP-07 mechanism)
  turbospark-guide/
    SKILL.md (index + config guide)
    diagnose-hooks/SKILL.md
    diagnose-commands/SKILL.md
    diagnose-skills/SKILL.md
    diagnose-mcp/SKILL.md
    diagnose-plugins/SKILL.md
  visualize/
    SKILL.md (output contract + chart guidance)
        |
        v
SkillManager discovery (unremovable, disable-able)
        |
        v
Agent activation via Skill tool
  diagnosis: read stores via existing file tools
  visualize: write artifact -> ArtifactWebView
```

## Alternatives Considered

### Alternative A: docs-page-only (no skills)

* **Pros**: no duplication.
* **Cons**: the agent cannot be guided to self-diagnose; users get
  generic help.
* **Reason for Rejection**: Rejected by author: the agent is the
  support surface.

### Alternative B: bundle d3-style vendor assets for visualize

* **Pros**: richer charts.
* **Cons**: bundle weight and license surface; ZP-06 already deferred
  vendor bundling; self-contained SVG covers the common cases.
* **Reason for Rejection**: Rejected by author: consistent with ZP-06.

## Non-Functional Requirements

* **Accuracy**: skill text must match real store paths and pane names;
  a test asserts each path string referenced by the guide exists in the
  app sources (docs-drift guard).
* **Security**: the visualize contract forbids external CDN loads by
  default (ArtifactWebView is offline-until-granted already; the skill
  must not instruct requests that trigger grants silently).
* **Localization**: skill content agent-facing English; listing strings
  localized.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/turbospark-guide/SKILL.md` (Create)
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/turbospark-guide/diagnose-hooks/SKILL.md` (Create)
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/turbospark-guide/diagnose-commands/SKILL.md` (Create)
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/turbospark-guide/diagnose-skills/SKILL.md` (Create)
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/turbospark-guide/diagnose-mcp/SKILL.md` (Create)
6. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/turbospark-guide/diagnose-plugins/SKILL.md` (Create)
7. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/visualize/SKILL.md` (Create)
8. `swift/TurboSparkApp/Sources/TurboSparkApp/State/SkillManager.swift` (Modify)
   * **Type / Scope**: discovery of the bundled root entries.
9. `swift/TurboSparkApp/Tests/TurboSparkAppTests/BundledGuideSkillsTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: discovery; every store path referenced by
     guide text exists (drift guard); visualize contract clauses present
     (size cap, no-CDN default); activation through the Skill tool.
   * Mutation checks: rename a referenced path fixture and assert the
     drift test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the guide tests.
* **Manual**: ask the agent to diagnose a deliberately broken MCP
  config; ask it to visualize a small dataset and confirm the artifact
  renders and obeys the no-CDN contract offline.

--------------------------------------------------------------------------------

## Implementation Notes

Implemented through ZP-07's bundled-skill root
(`swift/TurboSparkApp/Sources/TurboSparkAppResources/BundledSkills/`, a `.copy`
resource read by `SkillManager.discoverBundledSkills()`); ZP-05 builtin
marketplace seeding is only a later re-delivery path. Deviations from the
draft above, each from the EGM review in
`docs/egm-reviews/zcode-parity/ZP-20/`:

* **Two skills, not seven (review M5).** `scanDirectory` reads one folder
  level, so a nested `turbospark-guide/diagnose-hooks/SKILL.md` would never be
  discovered. The guide is one `turbospark-guide` skill with a section per
  diagnosis (hooks, commands, skills, MCP, plugins) plus `visualize`. This
  also keeps the listing cost at two entries. A test fails if a skill is
  nested below another skill folder.
* **Skills, not plugins (M2).** Only skills ship; the title word "plugins"
  means bundled content. Site publishing and legacy session restore (the other
  two thirds of finding 9.14) are inapplicable to TurboSpark. ZP-90 has no
  9.14 entry, so the exclusion is recorded here.
* **Guide source of truth: option (a), hand-written (M3).** The drift guard
  covers store FILE NAMES and Settings pane names only. Extraction rule: the
  last path component of every inline code span ending in `.json` must appear
  as a quoted literal in `Sources/TurboSparkApp`, and every `Settings > X`
  must be an `AppSettingsView.SettingsTab` title. It does not prove the
  surrounding prose is correct. Paths and panes the guide cites: app store
  `settings.json`, `projects_archive.json`, `global_mcp_servers.json`,
  `disabled_items.json`, `Hooks/trusted_hook_hashes.json`,
  `Hooks/custom_hooks.json`, `~/.turbospark/plugins/*.json`, user and project
  `skills/`, and the panes Hooks & Lifecycle, MCP Servers, Skills, Plugins,
  Profiles, Keyboard Shortcuts, General.
* **No MCP log (M8).** The app keeps no MCP log file, so the diagnosis path is
  the "Test connection and discover tools" result on the server card, which the
  skill asks the user to read back. The draft's "log output" is dropped.
* **Visualize contract is advisory (M4).** The 1 MB cap is not enforced in
  code and the skill says so. "Fragment-free" means a complete document; "no
  CDN by default" means no external loads at all, because the first blocked
  load shows the panel's "Network is off for this preview" banner whose Allow
  button is the user's grant, and the skill forbids designing for it. "Save a
  copy" exists in the artifact panel (`Generation/ArtifactPanelView.swift`).
  No validator was added (rejected as scope growth).
* **Security (M7).** The guide says to redact MCP `env`, `headers` and tokens,
  to read only what is needed, and adds no tools. Reads outside the project
  keep their normal permission prompts.
* **Disable-able is state-only.** A user switch-off persists through
  `SkillManager.setSkillEnabled` (`disabled_items.json`) and hides the skill
  from the Skill tool, tested. The Settings > Skills pane does not list
  bundled skills (it lists user, project and plugin skills), so there is no
  toggle for them yet. Deferred, same gap as `plugin-creator` (ZP-07).
* **Files.** The draft's `Resources/bundled-skills/` paths became
  `BundledSkills/` (the ZP-07 root); `SkillManager.swift` needed no change.
  Tests are in `BundledGuideSkillsTests.swift`. No localization keys were added:
  skill text is agent-facing English and the existing `[Bundled]` listing label
  is unchanged.

--------------------------------------------------------------------------------

*Drafted with EGM*
