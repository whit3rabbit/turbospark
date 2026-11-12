# ZP-20: Bundled content plugins (portable half of finding 9.14)

Finding 9.14 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md), portable half.
Parity targets: ZCode's
[zcode-guide plugin](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/zcode-guide-plugin/package.json)
and the visualize skill's
[output contract](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/visualize-plugin/skills/visualize/SKILL.md)
(related finding 9.9, Partial).

* Author: whit3rabbit
* Status: Draft
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

*Drafted with EGM*
