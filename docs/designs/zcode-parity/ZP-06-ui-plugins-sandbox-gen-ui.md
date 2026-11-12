# ZP-06: UI plugins, page sandbox, and Gen UI

Findings 4.3, 4.4, and 4.5 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[UI_PLUGIN.en.md](https://github.com/zai-org/ZCode/blob/main/UI_PLUGIN.en.md),
[pluginSandbox contract](https://github.com/zai-org/ZCode/blob/main/packages/desktop/src/main/pluginSandbox/CONTRACT.md),
and the feat/ui-plugin branch
([662c30be](../../ZCODE_COMPARISON.md), audited 2026-10-31). ZCode
revision `a60ba6a7` (v3.17.1), audited 2026-10-31
([INDEX.md](INDEX.md)).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

Terms: ZCode and the ZP set are defined in [INDEX.md](INDEX.md);
DESIGN.md is the app UI rule set; `make design-check` is its static scan;
a mutation check breaks the guard on purpose and confirms the test fails;
the ladder is the tool risk ladder the permission engine uses [author to
link the defining doc].

--------------------------------------------------------------------------------

## Overview

Let a plugin declare interactive web surfaces: HTML pages rendered inline
in the conversation or in a session side pane, backed by a strict page
sandbox, with Gen UI: the agent writes a small interactive HTML page (for
example sliders and charts) for one question and the chat renders it
inline.
TurboSpark already has `ArtifactWebView`
(`swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ArtifactWebView.swift`),
an offline-by-default sandboxed webview for model-written HTML artifacts;
this design generalizes that host into a plugin page system.

## The Problem

Plugin MCP results render as ordinary tool rows. A plugin can declare
`ui.surfaces` today, but `PluginManifestParser` ignores the field with
no diagnostic. Users get no interactive plugin pages,
no side pane, and the agent cannot produce rich interactive output
(sliders, charts, forms) beyond the existing artifact panel. ZCode ships
this whole layer (capability negotiation, page retention and teardown,
per-identity partitions, a `window.zcode`-style bridge) on Electron
guests; the Swift app has a webview host but no plugin UI layer.

## Goals

* Plugins can declare `ui.surfaces` entries; the manifest parser parses
  them into a typed model; a malformed entry produces a diagnostic.
* An enabled plugin's page opens inline (in the transcript as a card) or
  in a side pane, with a page lifecycle (retain while visible, tear down
  on close) and a provisional cap of 8 live pages (author to confirm;
  ZCode uses 64 on Electron, and TurboSpark shares unified memory with
  model weights). Opening past the cap closes the least recently visible
  page after teardown; a page being interacted with is never evicted.
  Page identity is plugin id + surface id + session id. The constant is
  `PluginUiHost.maxLivePages`.
* Each page identity gets an isolated `WKWebsiteDataStore` partition,
  derived `WKContentConfiguration`-style CSP from the manifest's resource
  declarations, Node-free, scriptable only from plugin-declared
  resources.
* A page bridge API, `window.turbospark`, exposes (v1): server tool calls
  (through the plugin's declared MCP tools), widget state get/set, and
  follow-up message send (draft-only: it fills the composer draft and
  never sends). `displayMode` is deferred until a CUJ needs it.
* Gen UI: the agent can write an HTML page with adjustable controls such
  as sliders that renders inline in the conversation using the same
  host, partition, and bridge. Gen UI pages are self-contained HTML (see
  Open Questions) and use the Gen UI page's own data store.

## Phasing

* **Phase 1: Gen UI.** Extend `ArtifactWebView` with the page bridge
  (widget state and draft-only follow-up). No manifest, no registry.
* **Phase 2: plugin surfaces.** Manifest model, `PluginUiHost`, side
  pane. Starts only when a real plugin declares a surface.

Target Files items 1, 5, 6, 8 move to phase 2. [Author to confirm.]

## Non-Goals

* No remote page loading from arbitrary URLs; pages come from plugin
  bundles or agent-written artifacts on disk.
* No script runtime of its own beyond the injected bridge; the bridge is
  the only app-provided script surface.
* No cross-page messaging.

## Critical User Journeys

### CUJ 1: plugin page in the side pane

Content source: the plugin ships page HTML in its bundle; `resourceUri`
is a bundle-relative path, never fetched over the network or from the MCP
server at render time [author to confirm]. A tool result references a
page with `_meta.ui.resourceUri` (as in ZCODE_COMPARISON 4.3); the
transcript row shows an Open button that opens the declared placement.
Example manifest entry [author to confirm field names]:

```json
{ "ui": { "surfaces": [ { "id": "chart", "resourceUri": "ui/chart.html",
    "resources": ["ui/chart.js"], "capabilities": ["invokeTool", "widgetState"],
    "placement": "sidePane" } ] } }
```

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
  follow-up send fills the composer draft, labeled with the page title
  and plugin name, for the user to send (draft-only, never auto-sent).

### CUJ 3: refused capability

* **Given**: a plugin page is open and its manifest does not declare
  `invokeTool`.
* **When**: the page calls `window.turbospark.invokeTool(...)`.
* **Then**: the bridge refuses with a named error visible in the page
  and logs; the page keeps rendering; nothing executes undeclared.

## Technical Plan (Architecture)

```text
PluginManifestParser: ui.surfaces typed model
  {resourceUri, resources[], capabilities[], placement?}
       |
       v
PluginUiHost (new)  <----  GenUiController
  page registry (identity -> live page),   agent tool: write page
  cap PluginUiHost.maxLivePages            (bounded size, app-generated name)
  placement: inline card or side pane      render inline via PluginUiHost
  lifecycle: retain while visible,         with the Gen UI identity
  teardown on close
       |
       v
ArtifactWebView (modified, optional bridge and CSP parameters)
  WKWebsiteDataStore nonPersistent() per identity
  CSP derived from manifest resource declarations
  navigation + permission gating
       |
       v
PageBridge (script message handler)
  invokeTool (declared tools only -> permission engine)
  sendFollowUp (draft-only, to composer)
  widgetState get/set (session-scoped store)
```

The existing `ArtifactWebView` grants model (offline by default,
per-content network grant baked into view identity, ephemeral storage)
stays the security floor. A plugin page has no more network or file
authority than an equivalently granted artifact. The bridge adds
script-to-app methods, each validated against the manifest.

## Alternatives Considered

### Alternative A: one shared WKWebsiteDataStore for all plugin pages

* **Pros**: no per-page store to allocate or free.
* **Cons**: pages from different plugins would share cookies and
  localStorage if the store were persistent, and one page could read
  another's data. A non-persistent store shared across views still
  lets two live pages observe each other's storage in memory.
* **Reason for Rejection**: `ArtifactWebView` already gives every view
  its own `.nonPersistent()` store. Keeping that per-view isolation
  costs nothing and is the existing security floor.

### Alternative B: native SwiftUI rendering of a declarative widget schema instead of HTML

* **Pros**: no web host; matches the native posture.
* **Cons**: not compatible with plugins that ship MCP Apps HTML, and
  it removes agent-authored Gen UI.
* **Reason for Rejection**: HTML is the interchange format of the
  ecosystem being adopted (ZCODE_COMPARISON.md 4.3). [Author to
  confirm: no recorded decision was found.]

### Alternative C: extend `ArtifactWebView` with a bridge only, no plugin layer

* **Pros**: smallest change, one security-critical host.
* **Cons**: no manifest-declared surfaces or side pane.
* **Reason for Rejection**: Not rejected. This is phase 1 (see Phasing).
  [Author to confirm.]

### Alternative D: defer plugin surfaces, ship Gen UI only

* **Pros**: Gen UI needs no manifest, registry or partitions.
* **Cons**: leaves finding 4.3 open.
* **Reason for Rejection**: Adopted as the phase order, not an
  exclusion. [Author to confirm.]

## Open Questions

* **Inline card vs side pane default**: Resolved per surface type: side
  pane for tool-result pages, inline for Gen UI answers. [Author to
  confirm.]
* **Gen UI vendor bundle**: Resolved: option (b), Gen UI pages are
  self-contained HTML; a bundled charts helper is a follow-up. [Author
  to confirm.]

## Non-Functional Requirements

* **Security**: the page sandbox denies all navigation and permissions by
  default; CSP is minimum-privilege from declarations; the bridge
  validates every call against the declaring manifest; instance cap is
  `PluginUiHost.maxLivePages`. Pages are offline and navigation is
  denied. A plugin page has no more authority than an equivalently
  granted artifact.
* **Privacy**: partitions are non-persistent; widget state is
  session-scoped and cleared on teardown.
* **Performance**: pages render off the transcript's main render path;
  teardown frees the data store; the cap bounds memory.
* **Design compliance**: the side pane, cards, and controls follow
  DESIGN.md; the sandbox contract itself gets a section in
  `swift/docs/SWIFT_PLUGINS.md`, like `SWIFT_GHOST_MODE.md` does for its
  subsystem [author to confirm model page].

## Risks and Mitigations

* **`sendFollowUp` as an injection channel.** A page can put text in
  front of the model. Mitigation: it only fills the composer draft,
  never sends; the draft is labeled with the page title and plugin
  name; the user presses send. [Author to confirm draft-only.]
* **`invokeTool` from a page.** A page can request tool calls in a
  loop. Mitigation: every call goes through the permission engine
  and the prompt names the page and plugin; calls need a recent user
  gesture in that page or are refused; at most N pending calls per
  page, rate limited to M per minute. [Author to set N and M.]
* **Gen UI has no manifest.** Mitigation: Gen UI pages get
  `widgetState` only. No `invokeTool`, `sendFollowUp` is draft-only,
  no network grant. Ladder position: render is not "low" until the
  page carries no tool authority.
* **Frames.** Bridge handlers accept messages from the main frame
  only (check `WKScriptMessage.frameInfo.isMainFrame`) and subframes
  are denied navigation. [Author to confirm iframes are unsupported
  in v1.]
* **Hostile page content.** All page content is untrusted; the
  bridge validates every call against the declaring manifest and
  fails closed.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/UI/PluginUiHost.swift` (Create)
   * **Type / Scope**: page host and registry.
   * **Proposed Changes**: page identity registry with the page cap
     (`PluginUiHost.maxLivePages`),
     placement routing (inline card vs side pane), lifecycle retention
     and teardown, partition allocation per identity
     (`WKWebsiteDataStore nonPersistent()`).
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ArtifactWebView.swift` (Modify)
   * **Type / Scope**: page webview.
   * **Proposed Changes**: add optional bridge and CSP parameters to the
     existing view instead of a sibling type. No network grant for plugin
     pages in v1; pages are offline.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/UI/PageBridge.swift` (Create)
   * **Type / Scope**: script-message bridge.
   * **Proposed Changes**: `window.turbospark` bridge (v1: invokeTool,
     widgetState get/set, sendFollowUp draft-only; displayMode is
     deferred until a CUJ needs it), per-call validation against the
     declaring manifest's capabilities, named refusal errors visible to
     the page (`undeclared_method`, `permission_denied`,
     `payload_too_large`, `rate_limited` [author to confirm names]), bridge
     messages at most [N] KB, TBD by author. Widget state is keyed by
     page identity, so no page reads another's state. Ladder per method:
     invokeTool asks per the tool's own ladder; widgetState is low;
     sendFollowUp is medium, draft-only.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Plugins/UI/GenUiController.swift` (Create)
   * **Type / Scope**: Gen UI agent tools.
   * **Proposed Changes**: write-page tool takes content only (at most
     [N] KB, TBD by author); the app generates the filename inside the
     chat's artifact folder; reuse the existing artifact write and
     registration path with a render-inline flag [author to confirm the
     registrar symbol]; render tool; registers through the ordinary tool
     registry with ladder entries. Ladder: write is medium risk, render
     is low only for pages with no tool authority (see Risks and
     Mitigations).
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/PluginManifestParser.swift` (Modify)
   * **Type / Scope**: manifest model.
   * **Proposed Changes**: parse `ui.surfaces` into a typed model
     (resourceUri, resources, capabilities, placement); keep a diagnostic
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
   * **Proposed Changes**: Route the side pane through the artifact panel
     (`ArtifactPanelView.swift`), not `SplitChatPanes.swift`, whose panes
     are read-only chat followers. [Author to confirm.] First consumer: a
     test plugin shipped under the app's fixtures. Lifecycle: page is
     torn down on pane close, chat switch and session end; widget state
     is cleared with it.

     | Event | Page | Widget state |
     | --- | --- | --- |
     | Pane close | torn down | cleared |
     | Chat switch | torn down | cleared |
     | App quit | torn down | cleared |
9. `swift/TurboSparkApp/Tests/TurboSparkAppTests/PluginUiHostTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: cap enforcement, teardown frees the slot,
     bridge validation (undeclared method refused), CSP derivation from
     declarations, Gen UI write bounds and path sanitization. Extend
     `ArtifactWebViewSecurityTests` with pure-function tests for:
     navigation to http(s), file outside folder and subframes refused;
     bridge rejects messages whose frame is not the main frame; bridge
     refuses undeclared methods and oversize payloads. Mutation checks
     are listed in the Verification Plan.
10. `swift/docs/SWIFT_PLUGINS.md` (Modify)
    * **Type / Scope**: docs.
    * **Proposed Changes**: new UI plugins section: manifest fields,
      sandbox contract, bridge API table, Gen UI contract, and the
      page-cap constant `PluginUiHost.maxLivePages`.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for PluginUiHostTests plus existing
  plugin manifest tests, and the extended `ArtifactWebViewSecurityTests`
  (navigation, frame and bridge policy tests, item 9).
* **Mutation checks**: drop the cap check and assert the cap test fails;
  loosen bridge validation and assert the refusal test fails; drop the
  main-frame check in the bridge and assert the frame test fails.
* **Static Analysis and Linting**: `make design-check` and
  `make design-check-full` for the card and side pane.
* **Localization**: the [ten parity gates](../../../swift/docs/SWIFT_LOCALIZATION.md);
  all new strings in the catalog for the
  [21 languages](../../../swift/docs/SWIFT_LOCALIZATION.md).
* **Manual**: load a test plugin declaring one surface, open the side
  pane, exercise invokeTool with allowed and undeclared methods, render
  a Gen UI page from a chat, close panes and confirm teardown, and
  verify a page cannot navigate to any network origin without the
  explicit grant.

--------------------------------------------------------------------------------

*Drafted with EGM*