# Critic report: ZP-06 (Stage 7)

Target: docs/designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md (lines 1-246). One round, no remediation. Reviewers: egm-design (DC, standard feature-spec-standard), egm-structural (ST, feature-spec-standard), egm-style (SY, no standard). All three completed.

## Design (DC)

Blocker
DC-1. Lines 139-155. Alternatives fail the Alternatives Rule: both "Rejected by author" with no link; only two strawmen; Alternative A is technically off (a shared default data store is persistent, a different question from per-identity non-persistent stores). Missing options: reuse ArtifactWebView with only a bridge added, defer Gen UI, WKURLSchemeHandler origin, per-origin partitioning. Fix: concrete technical reasons or a decision link, and add the missing alternatives.
DC-2. Lines 36-62, 104-137, 157-168. Security model has no threat analysis: `sendFollowUp` prompt injection (auto-send undefined, "attributed" undefined); permission prompt does not say a page originated the call, "allow always" and loop abuse unaddressed; Gen UI identity has no capability set (no manifest), so `invokeTool` could reach any tool; no sender-frame or origin check, iframes not addressed. Fix: add Risks and Mitigations and define the Gen UI capability set, defaulting to no `invokeTool`.

Major
DC-3. Lines 41-46, 112-114, 164-165. No memory budget for 64 WKWebViews on an app holding a large model; 64 only "matches ZCode". Fix: per-page budget, lower default cap, eviction policy.
DC-4. Lines 43-45, 107-111. Behavior at page 65 undefined; "retain while visible" can outlast the cap. Fix: specify policy and user-visible error.
DC-5. Lines 41-46, 177-178, 43. `nonPersistent()` returns a fresh store per call, so "partition per identity" and "frees the partition slot" are not WebKit concepts; "WKContentConfiguration-style CSP" is not a real API. Fix: name the mechanism (WKURLSchemeHandler with CSP header) and correct terms.
DC-6. Lines 58-60 vs 182-186, 134-137. Non-goals conflict with the network-grant flow; "fewer capabilities than a granted artifact" contradicts offering the same grant; CSP connect-src and grant scope unspecified. Fix: drop network grant for v1 or specify grant and CSP semantics.
DC-7. Lines 59-61, 163. Non-goal "no JS outside the webview" is not meaningful; "session-scoped" widget state plus "no cross-page messaging" is ambiguous. Fix: key widget state by page identity.
DC-8. Lines 64-73, 52-53. Open questions decide load-bearing items (placement default, vendor bundle); option (b) makes the vendor-resource goal dead. Fix: decide both, drop the vendor goal.
DC-9. Lines 25-35, 36-55, 154-155. No success metrics, no migration for plugins already declaring `ui.surfaces`, MCP Apps compatibility claimed but not mapped (`ui://`, JSON-RPC over postMessage). Fix: name the spec version and whether the bridge is wire-compatible.
DC-10. Lines 193-198, 192. Ladder positions unjustified; render executes agent script so "low" is wrong; write size and message bounds have no numbers. Fix: numeric limits, raise render.
DC-11. Lines 230-243, 222-223. No adversarial sandbox tests (file:, javascript:, data: navigation, window.open, iframes, CSP bypass, bridge origin spoofing); mutation checks cover two behaviors. Fix: automated escape tests and mutation checks for CSP and isolation.

Nit
DC-12. Lines 3-6, 12. Upstream links and commit hash lack a verification date. Fix: add it.
DC-13. Lines 34-35. "Swift app has a webview host" without a file path. Fix: cite the file.
DC-14. Lines 170-229. No sequencing or phasing across ten files. Fix: phases with checkpoints.
DC-15. Lines 166-168, 224-228. Page-cap constant location not given. Fix: name it.

## Structural (ST)

Major
ST-1. Lines 146-147, 154-155. "Rejected by author" with no link. Fix: concrete trade-offs or link. (Merges with DC-1.)
ST-2. Document-wide. No Risks and Mitigations section (required by the standard); NFRs list controls, not failure modes (sandbox escape, CSP bug, message flood, partition leak, agent HTML injection). Fix: add H2 after Non-Functional Requirements.
ST-3. Lines 98-101. CUJ 3 has Given and Then but no When. Fix: add "a page calls an undeclared bridge method".

Nit
ST-4. Lines 64-73. Open Questions precede the sections they depend on; Q1 has no owner. Fix: move after Alternatives and name an owner.
ST-5. Lines 225-228. Item 10 sub-bullets indented 3 spaces may not nest in CommonMark. Fix: indent 4 spaces.
ST-6. Lines 222-223. "Mutation checks" lacks the bold label used by sibling fields. Fix: label or move to Verification Plan.
ST-7. Lines 3-8. Commit "662c30be, 426 files" is bare text, not linked. Fix: link it.
ST-8. Lines 56-62, 44-46. Second non-goal overlaps the "Node-free" goal. Fix: reword or move to Security NFR.

Compliance notes: heading hierarchy clean; required sections present; INDEX constraints for tool ladder position and design-check and localization gates are met.

## Style (SY)

Major (all CALIBRATE unless noted)
SY-1. Lines 31-35. Problem claim not tied to a source or file path. Fix: cite ZCODE_COMPARISON finding numbers and the ArtifactWebView path.
SY-2. Lines 4-7. ZCode revision differs from INDEX pin (`a60ba6a7` / v3.17.1); revision for the "matching ZCode" claims unstated. Fix: state the revision checked.
SY-3. Lines 41-43, 56, 150-153, 165. 64 unjustified for a local-first app. Fix: mark provisional or estimate per-page memory.
SY-4. Lines 42-46. `WKContentConfiguration` is not a CSP API; "-style" hides an undefined mechanism; "Node-free" unclear. Fix: name the real mechanism, drop "Node-free".
SY-5. Lines 44-46. "scriptable only from plugin-declared resources" ambiguous. Fix: "executes only scripts loaded from resources listed in the manifest".
SY-6. Lines 64-73 vs 51-53. Goals promise bundled vendor resources, Open Question leans to none. Fix: reconcile.
SY-7. Lines 49, 57-58, 184. "Everything not declared is refused" and the network grant vs "no remote page loading": fetch/XHR permission unstated. Fix: state what the grant permits.
SY-8. Lines 134-140. "Fewer capabilities, never more" is false as written: the bridge adds `invokeTool` and `sendFollowUp`. Fix: narrow to WebKit-level capabilities.
SY-9. Lines 56-62. Non-goal about no app script runtime is unmeasurable and contradicts the injected bridge. Fix: "the app executes no page JavaScript in its own process", or drop.
SY-10. Lines 188-192. Size bounds have no numbers; tests cannot assert them. Fix: byte limits or TBD with owner.
SY-11. Lines 195-198. Ladder positions for `invokeTool` and `sendFollowUp` not stated. Fix: place each bridge method.
SY-12. Lines 221-223. Mutation checks phrased as verification fact inside the unit-test bullets. Fix: RELOCATE to Verification Plan.
SY-13. Lines 237-238. "ten parity gates" and "21 languages" unsourced. Fix: cite standard or drop counts.

Nit
SY-14. Lines 17-19, 49. "Gen UI as the agent-facing form" and "Tweak-style controls" are jargon. Fix: plain wording or define.
SY-15. Lines 21-23. "without weakening its grants model" is an outcome claim. Fix: drop or move to NFR.
SY-16. Line 29. Diagnostic string not quoted. Fix: quote or cite parser line.
SY-17. Line 144. "exist for exactly this" speculates about ZCode motive. Fix: drop.
SY-18. Line 143. "compromised page reads another's storage" lacks condition. Fix: say a shared store shares cookies and localStorage per origin.
SY-19. Lines 146-147, 154-155. Rationale restated. Fix: shorten.
SY-20. Lines 49, 117, 160. Deny-by-default stated three times. Fix: say once in NFR.
SY-21. Line 167. "like the other subsystems" vague. Fix: name one contract page.
SY-22. Lines 49, 189. "Facade" and "bridge" used interchangeably. Fix: use "bridge".
