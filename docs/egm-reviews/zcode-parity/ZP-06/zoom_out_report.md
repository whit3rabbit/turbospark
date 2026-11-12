# Zoom out report: ZP-06 (Stage 5)

Target: docs/designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md (full file, 246 lines)
Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.
Limitation: both reviewers read ZCODE_COMPARISON.md only through line 237, so findings 4.3 to 4.5 were not read in full.

## Zoom out (ZO)

ZO-1. Major, lines 20-23, 47-54, 126-132, 134-137. Three findings in one design with no staging. Gen UI does not need manifests, a 64-page registry, per-plugin partitions or an `invokeTool` bridge. Fix: phase 1 Gen UI on the existing artifact panel, phase 2 plugin surfaces gated on a real plugin that declares a surface.

ZO-2. Major, lines 193-198 and 36-54. Gen UI controller and its two tools duplicate `Tools/File/ArtifactRegistrar.swift`, `FileReadWriteTools.swift`, `ArtifactPanelView` and `State/AppArtifact.swift`. Fix: reuse the artifact write/register path, add at most a render-inline flag, drop the separate write tool and `GenUiController`.

ZO-3. Major, lines 134-137, 180-186. "Generalize ArtifactWebView" as a sibling `PluginPageWebView` risks forking the security-critical sandbox config; "fewer capabilities, never more" is unenforced. Fix: extend `ArtifactWebView` with bridge and CSP parameters, one sandbox implementation.

ZO-4. Major, lines 43-46, 106-112, 174-179. Per-identity partition plus 64-page cap is copied from Electron. `.nonPersistent()` is already per view; registry only matters if hidden pages are retained, which is not stated. Fix: one live page per pane or card, teardown on disappear, no registry, or justify 64 with a measurement.

ZO-5. Major, lines 213-216. `SplitChatPanes.swift` holds read-only chat panes; a plugin page there mixes pane kinds. `ArtifactPanelView` is the existing non-chat side surface. Fix: route plugin pages through the artifact panel or a sibling, or justify the extension.

ZO-6. Major, lines 38-40, 127, 199-204. Manifest model and capability schema are specified for a surface no plugin in the repo declares. Fix: name the first concrete plugin needing a page and size the model to it.

ZO-7. Major, lines 47-52, 120-125. Bridge is broader than any CUJ: `displayMode` is unused; `sendFollowUp` from a model-authored page is a prompt-injection channel. Fix: ship `invokeTool` first, defer `displayMode`, require user confirmation for `sendFollowUp`, state the threat in the NFR.

ZO-8. Major, lines 64-73. Open questions contradict the plan: the plan builds both placements, and the vendor-resource goal (lines 51-54) is dead scope under lean (b). Fix: resolve (b), drop the vendor goal, pick one placement for phase 1.

ZO-9. Nit, lines 4-7, 47. "Parity" is not testable; which ZCode contract rules are adopted is not stated. Fix: add an adopted/changed/dropped table.

ZO-10. Nit, lines 106-132. Architecture block mixes layers; `GenUiController` has no arrow to `PluginUiHost`, one arrow points up. Fix: redraw linear: manifest, host, webview, bridge, with Gen UI as a second entry.

ZO-11. Nit, lines 170-228, 217-223. No automated test for navigation or permission denial despite the security claims at 159-165. Fix: add tests mirroring any ArtifactWebView tests.

ZO-12. Nit, line 197. "Medium" and "low" risk are not placed on a named ladder as INDEX.md (lines 121-122) requires. Fix: name `AppToolPermissionEngine` entries or drop the tools per ZO-2.

Reuse note: `DOMSnapshotService` (BrowserAutomation) is an existing `WKScriptMessageHandlerWithReply` to model the bridge on.

## Inventive catalyst (IC)

IC-1. Major, lines 41-43, 108, 161-165, 174-179. Cap of 64 borrowed from Electron; each WKWebView is a heavy process next to local model weights; behavior at page 65 undefined. Fix: small live cap with LRU snapshot-and-evict, and specify behavior at the cap.

IC-2. Major, lines 51-53, 94-96, 129-131, 160-161, 187-192. Gen UI pages have no manifest, so the bridge "validates against the declaring manifest" has no capability source for the most dangerous surface; `sendFollowUp` auto-send is a self-prompting loop; "render is low" understates risk. Fix: Gen UI capability subsection (widgetState and displayMode only, no `invokeTool`, `sendFollowUp` draft-only), raise render to at least medium.

IC-3. Major, lines 83-87, 120-125, 187-192. Page-initiated `invokeTool` can loop without a user gesture, spamming approval prompts. Fix: rate limit, pending-call cap, user-activation requirement, attribution label in the prompt.

IC-4. Major, lines 47-50, 149-155, 187-192. Bespoke `window.turbospark` bridge reintroduces the incompatibility used to reject Alternative B. Fix: state which spec the bridge conforms to (MCP Apps JSON-RPC over postMessage) and make `window.turbospark` a shim.

IC-5. Major, lines 17-19, 43-46, 56-59, 79-85, 159-160. Page content source unspecified (resourceUri implies MCP-fetched HTML, line 58 says on-disk only); "WKContentConfiguration-style CSP" is not a real WebKit API and file loads cannot carry a CSP header. Fix: serve through a WKURLSchemeHandler that injects CSP, and drop the nonexistent API name.

IC-6. Major, lines 17-23, 52-53, 70-73, 170-228. One system, two products; vendor bundle is both Goal and undecided. Fix: two phases with an exit gate; vendor bundle a non-goal for phase 1.

IC-7. Major, lines 98-102, 159-165. Only undeclared-method failure handled; no web content process termination, hang or load-failure handling. Fix: add a crash and hang row to the lifecycle spec, auto-reload once, and a test.

IC-8. Nit, lines 134-137 vs 182-186. "Fewer capabilities than a granted artifact" contradicts file 2 reusing the grant flow (equal). Fix: reword to "no more than an equivalently granted artifact".

IC-9. Nit, lines 77-87, 161-163. "Session-scoped" widget state and behavior on chat switch or under ZP-23 undefined. Fix: lifecycle table keyed to chat selection.

IC-10. Nit, lines 130, 195-196. Sanitizing a model-supplied path is weaker than not accepting one. Fix: tool takes content only, generates filename in the chat artifact folder.

IC-11. Nit, lines 217-223, 239-243. Navigation denial and partition isolation only verified manually. Fix: automated tests with mutation checks.
