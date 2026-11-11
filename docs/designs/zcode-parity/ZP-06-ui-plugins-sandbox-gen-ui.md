# ZP-06: UI plugins, page sandbox, and Gen UI

Findings 4.3, 4.4, and 4.5 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[UI_PLUGIN.en.md](https://github.com/zai-org/ZCode/blob/main/UI_PLUGIN.en.md),
[pluginSandbox contract](https://github.com/zai-org/ZCode/blob/main/packages/desktop/src/main/pluginSandbox/CONTRACT.md),
and the feat/ui-plugin branch (662c30be, 426 files).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Let a plugin declare interactive web surfaces: HTML pages rendered inline
in the conversation or in a session side pane, backed by a strict page
sandbox, with Gen UI as the agent-facing form (the agent writes an
interactive HTML page per question and the conversation renders it).
TurboSpark already has `ArtifactWebView`, an offline-by-default sandboxed
webview for model-written HTML artifacts; this design generalizes that
host into a plugin page system without weakening its grants model.

## The Problem

Plugin MCP results render as ordinary tool rows. A plugin can declare
`ui.surfaces` today and the manifest parser records it as a one-line
"unsupported surface" diagnostic. Users get no interactive plugin pages,
no side pane, and the agent cannot produce rich interactive output
(sliders, charts, forms) beyond the existing artifact panel. ZCode ships
this whole layer (capability negotiation, page retention and teardown,
per-identity partitions, a `window.zcode`-style bridge) on Electron
guests; the Swift app has a webview host but no plugin UI layer.

## Goals

* Plugins can declare `ui.surfaces` entries; the manifest parser parses
  them into a typed model instead of a diagnostic.
* An enabled plugin's page opens inline (in the transcript as a card) or
  in a side pane, with a page lifecycle (retain while visible, tear down
  on close) and a cap on live pages (64, matching ZCode).
* Each page identity gets an isolated `WKWebsiteDataStore` partition,
  derived `WKContentConfiguration`-style CSP from the manifest's resource
  declarations, Node-free, scriptable only from plugin-declared
  resources.
* A page bridge API, `window.turbospark`, exposes: server tool calls
  (through the plugin's declared MCP tools), follow-up message send,
  widget state get/set, and display-mode change requests. Everything not
  declared is refused.
* Gen UI: the agent can write an HTML page with Tweak-style controls that
  renders inline in the conversation using the same host, partition, and
  bridge, with vendor resources (charts) bundled and mapped inside the
  partition.

## Non-Goals

* No remote page loading from arbitrary URLs; pages come from plugin
  bundles or agent-written artifacts on disk.
* No JavaScript execution outside the page's own webview context (the
  app adds no script runtime of its own).
* No cross-page messaging.

## Open Questions

* **Inline card vs side pane default**: ZCode supports both; which is the
  default for a `ui.surfaces` entry without a placement hint? Suggest
  side pane for tool-result pages, inline for Gen UI answers. The author
  decides.
* **Gen UI vendor bundle**: shipping d3-equivalents in the app bundle
  adds weight. Options: (a) bundle a minimal charts helper; (b) skill
  instructs the model to write self-contained HTML with no vendor deps;
  (c) both. Lean (b) day one.

## Critical User Journeys

### CUJ 1: plugin page in the side pane

* **Given**: an enabled plugin declares a surface with a resourceUri and
  declared resources.
* **When**: its MCP tool result carries the page reference and the user
  opens it.
* **Then**: the side pane loads the page in the plugin's partition with
  the derived CSP; the page calls `window.turbospark.invokeTool(...)`
  and the call runs the declared MCP tool through the ordinary
  permission engine; closing the pane tears the page down and frees the
  partition slot.

### CUJ 2: Gen UI question

* **Given**: the user asks a question with a visual component.
* **When**: the agent writes an HTML page via the Gen UI tool and calls
  the render tool.
* **Then**: an inline card renders the page in the Gen UI partition with
  widget state bridged; the user drags a slider, and the page's
  follow-up send posts an attributed message into the composer.

### CUJ 3: refused capability

* **Given**: a page calls a bridge method its manifest did not declare.
* **Then**: the bridge refuses with a named error visible in the page
  and logs; the page keeps rendering; nothing executes undeclared.

## Technical Plan (Architecture)

```text
PluginUiHost (new)
  page registry (identity -> live page), cap 64
  placement: inline card or side pane
  lifecycle: retain while visible, teardown on close
       |
       v
PluginPageWebView (generalized ArtifactWebView)
  WKWebsiteDataStore nonPersistent() per identity
  CSP derived from manifest resource declarations
  navigation + permission gating: everything denied
  except plugin-declared local resources
       |
       v
PageBridge (script message handler)
  invokeTool (declared tools only -> permission engine)
  sendFollowUp (attributed, to composer)
  widgetState get/set (session-scoped store)
  displayMode request
       ^
       | declares
PluginManifestParser: ui.surfaces typed model
  {resourceUri, resources[], capabilities[], placement?}
GenUiController
  agent tool: write page (bounded size, sanitized path)
  render inline via PluginUiHost with the Gen UI identity
```

The existing `ArtifactWebView` grants model (offline by default,
per-content network grant baked into view identity, ephemeral storage)
stays the security floor: a plugin page gets *fewer* capabilities than an
artifact with a granted network access, never more.

## Alternatives Considered

### Alternative A: render plugin pages in a shared process via WebKit's shared data store

* **Pros**: fewer partitions to manage.
* **Cons**: one compromised page reads another's storage; ZCode's
  per-identity partitions exist for exactly this.
* **Reason for Rejection**: Rejected by author: per-identity isolation
  is the security boundary.

### Alternative B: native SwiftUI rendering of a declarative widget schema instead of HTML

* **Pros**: no web host at all; matches the app's native posture.
* **Cons**: not compatible with the MCP Apps / plugin ecosystem shape
  (HTML pages are the interchange), and kills agent-authored Gen UI.
* **Reason for Rejection**: Rejected by author: HTML is the
  interchange format of the ecosystem being adopted.

## Non-Functional Requirements

* **Security**: the page sandbox denies all navigation and permissions by
  default; CSP is minimum-privilege from declarations; the bridge
  validates every call against the declaring manifest; instance cap 64.
* **Privacy**: partitions are non-persistent; widget state is
  session-scoped and cleared on teardown.
* **Performance**: pages render off the transcript's main render path;
  teardown frees the data store; the cap bounds memory.
* **Design compliance**: the side pane, cards, and controls follow
  DESIGN.md; the sandbox contract itself gets a `swift/docs/` contract
  page like the other subsystems.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/UI/PluginUiHost.swift` (Create)
   * **Type / Scope**: page host and registry.
   * **Proposed Changes**: page identity registry with the 64-page cap,
     placement routing (inline card vs side pane), lifecycle retention
     and teardown, partition allocation per identity
     (`WKWebsiteDataStore nonPersistent()`).
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/UI/PluginPageWebView.swift` (Create)
   * **Type / Scope**: page webview.
   * **Proposed Changes**: generalized from `ArtifactWebView`: derived
     CSP, deny-all navigation and permission delegates except declared
     local resources, offline by default (network only if the manifest
     declares it and the user grants it, reusing the artifact grant
     flow).
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/UI/PageBridge.swift` (Create)
   * **Type / Scope**: script-message bridge.
   * **Proposed Changes**: `window.turbospark` facade (invokeTool,
     sendFollowUp, widgetState, displayMode), per-call validation
     against the declaring manifest's capabilities, named refusal errors
     visible to the page, size bounds on messages.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/UI/GenUiController.swift` (Create)
   * **Type / Scope**: Gen UI agent tools.
   * **Proposed Changes**: write-page tool (bounded size, path
     sanitized to the chat's artifact folder) and render tool; registers
     through the ordinary tool registry with ladder entries (write is
     medium risk, render is low).
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/PluginManifestParser.swift` (Modify)
   * **Type / Scope**: manifest model.
   * **Proposed Changes**: parse `ui.surfaces` into a typed model
     (resourceUri, resources, capabilities, placement); replace the
     unsupported-surface diagnostic with real parsing; keep a diagnostic
     only for malformed entries.
6. `swift/TurboSparkApp/Sources/TurboSparkApp/State/PluginModels.swift` (Modify)
   * **Type / Scope**: model types.
   * **Proposed Changes**: surface/capability types and the live-page
     registry entry type.
7. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/TranscriptPluginCardView.swift` (Create)
   * **Type / Scope**: inline card UI.
   * **Proposed Changes**: inline placement host with page chrome
     (title, close, display mode), DESIGN.md compliant, localized.
8. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/SplitChatPanes.swift` (Modify)
   * **Type / Scope**: side pane routing.
   * **Proposed Changes**: add the plugin side pane target beside the
     existing read-only chat panes, with the pane lifecycle contract.
9. `swift/TurboSparkApp/Tests/TurboSparkAppTests/PluginUiHostTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: cap enforcement, teardown frees the slot,
     bridge validation (undeclared method refused), CSP derivation from
     declarations, Gen UI write bounds and path sanitization.
   * Mutation checks: drop the cap check and assert the cap test fails;
     loosen bridge validation and assert the refusal test fails.
10. `swift/docs/SWIFT_PLUGINS.md` (Modify)
   * **Type / Scope**: docs.
   * **Proposed Changes**: new UI plugins section: manifest fields,
     sandbox contract, bridge API table, Gen UI contract, and the
     page-cap constant.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for PluginUiHostTests plus existing
  plugin manifest tests.
* **Static Analysis and Linting**: `make design-check` and
  `make design-check-full` for the card and side pane.
* **Localization**: ten parity gates; all new strings in the catalog for
  21 languages.
* **Manual**: load a test plugin declaring one surface, open the side
  pane, exercise invokeTool with allowed and undeclared methods, render
  a Gen UI page from a chat, close panes and confirm teardown, and
  verify a page cannot navigate to any network origin without the
  explicit grant.

--------------------------------------------------------------------------------

*Drafted with EGM*