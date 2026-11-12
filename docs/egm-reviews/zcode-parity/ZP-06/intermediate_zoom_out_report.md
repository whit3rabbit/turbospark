# Intermediate zoom out report: ZP-06 (Stage 8)

Target: docs/designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md (full file). Reviewer: egm-zoom-out (S8). Completed. No remediation was applied, so the target is unchanged since Stage 5. Prior reports: zoom_out_report.md, comprehension_report.md, critic_report.md.

Code checks by the reviewer: `ArtifactWebView` exists at Generation/ArtifactWebView.swift:83 with `ArtifactWebViewSecurityTests.swift`; `DOMSnapshotService` is a usable `WKScriptMessageHandlerWithReply` model; no `WKURLSchemeHandler` exists in the Swift tree; the "unsupported surface" diagnostic is referenced at swift/docs/SWIFT_PLUGINS.md:189. The reviewer did not open `SplitChatPanes`, `ArtifactRegistrar`, `FileReadWriteTools` or `ArtifactPanelView`.

Stage 5 to 7 findings judged valid and in scope: staging (ZO-1, IC-6), Gen UI duplicates the artifact path (ZO-2), extend not fork (ZO-3, strongest simplicity finding), the cap of 64 (ZO-4, IC-1, DC-3, DC-4), bridge security (ZO-7, IC-2, IC-3, DC-2), dead vendor goal (ZO-8, DC-8, SY-6), CSP mechanism (IC-5, DC-5, CO-3, SY-4), "fewer, never more" (IC-8, DC-6, SY-8).

Judged valid but scope-expanding (demote or defer): ZO-5 (unverified), ZO-6 (product question), IC-4 (one sentence naming the spec), IC-7, IC-9, DC-9, DC-11, DC-14. Skip as Nits or scope growth: DC-12, DC-13, DC-15, SY-1, SY-2, SY-13, CO-10, ZO-9, ZO-11, ZO-12.

## Findings

S8-1. Major, lines 17-23, 47-54, 170-228. Three findings bundled with no phasing. Fix: phase 1 Gen UI on the extended ArtifactWebView; phase 2 plugin surfaces, gated on a real plugin that declares one.
S8-2. Major, lines 134-137, 180-186. A sibling `PluginPageWebView` risks forking the sandbox. Fix: extend ArtifactWebView with bridge and CSP parameters; reword "fewer, never more" to "no more than an equivalently granted artifact".
S8-3. Major, lines 193-198, 51-54, 94-96. `GenUiController` and its tools duplicate the artifact write/register path. Fix: reuse the registrar with a render-inline flag; content-only tool with generated filename; name ladder entries and raise render above "low".
S8-4. Major, lines 41-46, 106-112, 164-165, 177-178. Cap of 64 unjustified; "partition per identity" is not a WebKit concept; identity undefined; page 65 undefined. Fix: define identity, smaller provisional cap, stated policy at the cap, no registry unless hidden pages are retained.
S8-5. Major, lines 42-46, 58-60. Content source and CSP mechanism unspecified; "Node-free" is Electron residue. Fix: name a WKURLSchemeHandler with CSP injection, define the `resourceUri` source, drop "Node-free".
S8-6. Major, lines 47-52, 96, 120-125, 187-192. Bridge broader than any CUJ and its security model undefined. Fix: Gen UI gets widgetState only, `sendFollowUp` draft-only, defer `displayMode`, add rate limits and an attribution label in the permission prompt, add a Risks section.
S8-7. Major, lines 64-73 vs 51-53, 77-87. Open Questions contradict the plan. Fix: resolve (b), delete the vendor goal, pick the placement default per surface type.
S8-8. Major, lines 139-155. Alternatives weak; "Rejected by author" with no rationale. Fix: concrete trade-offs; add "extend ArtifactWebView with a bridge only" and "defer plugin surfaces".
S8-9. Major, lines 98-102. CUJ 3 has no When; page-reference format and how the user opens a page (CUJ 1) undefined. Fix: add the When and define the reference.
S8-10. Nit, lines 106-132. Upward "declares" arrow; `GenUiController` not connected to `PluginUiHost`. Fix: redraw linearly.
S8-11. Nit, lines 217-228, 239-243. No automated navigation, permission or partition-isolation tests. Fix: extend `ArtifactWebViewSecurityTests`; attach mutation checks to the Verification Plan; give numeric message and write bounds.
