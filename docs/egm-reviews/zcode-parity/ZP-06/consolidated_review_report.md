# Consolidated review report: ZP-06 UI plugins, page sandbox, and Gen UI

Target: docs/designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md (full file, 246 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed on the first try except egm-inventive-catalyst, whose first spawn was refused ("Concurrent subagent limit reached") and succeeded on one retry.
Source IDs: ZO and IC = zoom_out_report, CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.
Caveat: Stage 5 reviewers read ZCODE_COMPARISON.md only to line 237, so findings 4.3 to 4.5 were not checked against the source. SplitChatPanes fit (M13) was not verified in code.

Counts after deduplication: Blocker 3, Major 14, Nit 11.

## Blocker

B1. Lines 139-155. Alternatives fail the Alternatives Rule: both "Rejected by author" with no link; Alternative A mis-states the technical question (shared persistent store vs non-persistent store per view); missing "extend ArtifactWebView with a bridge only", "defer plugin surfaces or Gen UI", scheme-handler origin. Fix: concrete trade-offs or decision links, and add the missing alternatives. [DC-1, ST-1, S8-8, SY-17, SY-18, SY-19]
B2. Lines 36-62, 104-137, 157-168 (missing section). No threat analysis or Risks and Mitigations section. `sendFollowUp` is a prompt-injection and self-prompting channel (auto-send and "attributed" undefined); page-initiated `invokeTool` has no rate limit, user gesture or origin label in the permission prompt; Gen UI has no manifest so its capability set is undefined; no sender-frame or iframe policy. Fix: add Risks and Mitigations; Gen UI gets widgetState only, no `invokeTool`, `sendFollowUp` draft-only; add rate limit, pending-call cap and attribution label; raise render ladder above "low". [DC-2, ST-2, ZO-7, IC-2, IC-3, CO-7, CO-9, SY-11, S8-6, S8-3]
B3. Lines 42-46, 177-178, 159-160. The CSP mechanism is not real: "WKContentConfiguration-style CSP" is not a WebKit API and file loads cannot carry a CSP header; `nonPersistent()` returns a fresh store per call, so "partition per identity" and "frees the partition slot" are not WebKit concepts. Fix: serve all page content through a WKURLSchemeHandler that injects CSP and is the only origin navigation allows; define identity and reword the isolation claim. [IC-5, DC-5, CO-3, SY-4, S8-4, S8-5]

## Major

M1. Lines 17-23, 47-54, 170-228. Three findings bundled with no phasing; Gen UI does not need manifests, registry or plugin partitions. Fix: phase 1 Gen UI on the extended ArtifactWebView, phase 2 plugin surfaces gated on a real declaring plugin. [ZO-1, IC-6, DC-14, S8-1]
M2. Lines 134-137, 180-186. Sibling `PluginPageWebView` risks forking the security-critical sandbox; "fewer capabilities, never more" is false (bridge adds methods; same grant flow gives parity). Fix: extend ArtifactWebView with bridge and CSP parameters; reword to "no more than an equivalently granted artifact". [ZO-3, IC-8, DC-6, SY-8, S8-2]
M3. Lines 193-198, 51-54, 130. Gen UI controller and tools duplicate the artifact write/register path; model-supplied path is a weak control. Fix: reuse the registrar with a render-inline flag, content-only tool with generated filename, named ladder entries. [ZO-2, IC-10, DC-10, S8-3]
M4. Lines 41-46, 106-112, 164-165. Cap of 64 borrowed from Electron, no memory budget next to local model weights, "page identity" undefined, page 65 behavior undefined. Fix: smaller provisional cap, define identity, state refuse or evict policy. [ZO-4, IC-1, DC-3, DC-4, CO-2, SY-3, S8-4]
M5. Lines 17-19, 79-85, 56-59. Content source and flow unspecified: `resourceUri` implies MCP-fetched HTML, line 58 says on-disk only; the page reference in the tool result and how the user opens it are undefined; manifest fields have no example or vocabulary. Fix: define source, reference format, open action, and one example manifest. [IC-5, CO-4, CO-5, S8-9]
M6. Lines 64-73 vs 51-53, 77-87. Open Questions contradict the plan: option (b) kills the vendor-resource goal; both placements are built while the default is open. Fix: resolve (b), drop the vendor goal, pick placement per surface type. [ZO-8, DC-8, SY-6, CO-6, S8-7]
M7. Lines 56-62, 134-137, 182-186. Non-goals vs network grant: scope of grant (origins, fetch/XHR, CSP connect-src) unstated; non-goal "no JS outside the webview" is unmeasurable and contradicts the injected bridge. Fix: drop the grant for v1 or specify it; reword or move the non-goal to Security NFR. [DC-6, DC-7, SY-7, SY-9, CO-8, ST-8]
M8. Lines 98-101. CUJ 3 has Given and Then but no When. Fix: add "a page calls an undeclared bridge method". [ST-3, S8-9]
M9. Lines 47-52, 96, 120-125, 163. Bridge broader than any CUJ (`displayMode` unused); widget state scope unclear vs "no cross-page messaging"; display modes and error names not enumerated. Fix: ship `invokeTool` and widgetState first, key widget state by page identity, enumerate errors. [ZO-7, CO-7, DC-7]
M10. Lines 47-50, 149-155. Bespoke `window.turbospark` bridge vs the MCP Apps interchange claimed in Alternative B; no spec named. Fix: name the spec and whether the bridge is wire-compatible (shim over JSON-RPC postMessage). [IC-4, DC-9]
M11. Lines 217-223, 239-243. No automated tests for navigation or permission denial, partition isolation, iframes, bridge origin spoofing; mutation checks cover two behaviors and sit in the unit-test bullets. Fix: extend `ArtifactWebViewSecurityTests`, move mutation checks to the Verification Plan. [ZO-11, IC-11, DC-11, SY-12, ST-6, S8-11]
M12. Lines 98-102, 159-165. Failure handling is passive: no web content process termination, hang or load-failure path (likely under memory pressure). Fix: auto-reload once from stored page and state, retry chrome, and a test. Reviewer S8 suggested this may be deferred. [IC-7, S8 demote]
M13. Lines 213-216, 36-40. Side pane routed into `SplitChatPanes` (read-only chat panes) rather than the artifact panel; no first concrete plugin needs a surface; session lifecycle under chat switch and ZP-23 undefined. Unverified in code. Fix: justify or reroute via the artifact panel, name the first consumer, add a lifecycle table. [ZO-5, ZO-6, IC-9]
M14. Lines 188-192, 195-198, 237-238, 4-7, 31-35. Missing numbers and sources: message and write size bounds, ladder position per bridge method, "ten gates" and "21 languages", ZCode revision for parity claims (INDEX pins a60ba6a7), no source path for the "Swift app has a webview host" claim. Fix: give byte limits or TBD with owner, place each method on the ladder, cite or drop counts, state the revision and file path. [SY-1, SY-2, SY-10, SY-11, SY-13, DC-10, DC-13, CO-12]

## Nit

N1. Lines 3-8. ZCode commit "662c30be, 426 files" bare text, links lack a verification date; no adopted/changed/dropped parity table. Fix: link, date, table. [ST-7, DC-12, ZO-9]
N2. Lines 106-132. Architecture diagram: upward "declares" arrow, `GenUiController` not linked to `PluginUiHost`. Fix: redraw linearly with Gen UI as a second entry. [ZO-10, CO-13, S8-10]
N3. Lines 64-73. Open Questions placed before the sections they depend on; Q1 has no owner. Fix: move after Alternatives and name an owner. [ST-4]
N4. Lines 225-228. Item 10 sub-bullets indented 3 spaces may not nest. Fix: indent 4 spaces. [ST-5]
N5. Lines 17-19, 49. Jargon undefined: "Gen UI as the agent-facing form", "Tweak-style controls", "Gen UI partition". Fix: plain wording or define. [SY-14, CO-6]
N6. Lines 21-23, 29. "Without weakening its grants model" is an outcome claim; diagnostic string not quoted. Fix: move to NFR; quote or cite parser line. [SY-15, SY-16]
N7. Lines 143-144. "Compromised page reads another's storage" lacks its condition; "exist for exactly this" speculates about ZCode. Fix: state the shared-store condition, drop the motive. [SY-17, SY-18]
N8. Lines 49, 117, 160. Deny-by-default stated three times. Fix: say once in NFR. [SY-20]
N9. Lines 49, 189. "Facade" and "bridge" used interchangeably. Fix: use "bridge". [SY-22]
N10. Lines 166-168, 224-228. "Like the other subsystems" is vague; page-cap constant not named or located. Fix: name one contract page and the constant. [SY-21, DC-15]
N11. Undefined project terms: ZCode, MCP Apps, DESIGN.md, design-check, mutation checks, "permission engine", "ladder". Fix: link INDEX or one-line definitions. [CO-9, CO-10, CO-11]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-12 (zoom-out), IC-1 to IC-11 (inventive catalyst). Feeds B2, B3, M1 to M4, M6, M9 to M13, N1, N2.
- comprehension_report.md: CO-1 to CO-13. Feeds B2, B3, M4, M5, M7, M9, M14, N5, N11. CO-1 is covered by M2 and CO-8 by M7.
- critic_report.md: DC-1 to DC-15, ST-1 to ST-8, SY-1 to SY-22. Feeds all Blockers and Majors and N1 to N10.
- intermediate_zoom_out_report.md: S8-1 to S8-11. Confirms ZO-1 to ZO-4, ZO-7, ZO-8, IC-5, IC-8, DC-2 to DC-8 as valid and in scope; recommends deferring ZO-5, ZO-6, IC-4 as a mandate, IC-7, IC-9, and skipping citation-only and count-only Nits (SY-1, SY-2, SY-13, DC-12, DC-13, DC-15, CO-10, ZO-9, ZO-11, ZO-12). Those are kept here as M12 to M14 and Nits per the severity given by their source reviewers, with the S8 view noted.
- Merge notes: DC-1 and ST-1 merged into B1. DC-2 and ST-2 merged into B2. Severity of B3 is raised from the reviewers' Major to Blocker because the isolation and CSP claims are the design's security boundary and name a nonexistent mechanism.
