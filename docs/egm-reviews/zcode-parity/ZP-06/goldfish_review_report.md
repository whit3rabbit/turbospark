Status: resolved

## Applied

* Applied (25): B1, B2, M1, M2, M3, M4, M5, M6, M7, M8, M9, M11, M13, M14, N1-N11.
* Unapplied: B3 and M10 (ESCALATE), M12 (CONTEST), all skipped by instruction.
* Notes: B2's "change line 163" was applied to CUJ 2 and Goal 4 (draft-only), since line 163 is the Privacy bullet. The ladder column (M14) was written inline in Target Files item 3 because no bridge table exists in the doc. The summary table says PROPOSE FIX 24, but 25 findings are listed; no finding lacks a category.

# Goldfish Annotated Review Report

**Target:** `docs/designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md` (full file, 246 lines)
**Source findings:** `docs/egm-reviews/zcode-parity/ZP-06/consolidated_review_report.md` (28 findings: Blocker 3, Major 14, Nit 11)

## Summary

| Category | Count | Items |
| --- | --- | --- |
| PROPOSE FIX | 24 | B1, B2, M1-M9, M11, M13, M14, N1-N11 |
| CONTEST | 1 | M12 |
| ESCALATE | 2 | B3, M10 |
| DECLINE | 0 | none |

Severities are unchanged from the consolidated report. Nits are all PROPOSE FIX because each edit is one line.

## Verified-false or partly-false reviewer claims

* **M12 (no web content process termination path): false for the host this design generalizes.** `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ArtifactWebView.swift`, `ArtifactWebView.Coordinator.webViewWebContentProcessDidTerminate` reloads once (`shouldReloadAfterTermination`) and then loads `ArtifactWebView.stoppedHTML`. The design says `PluginPageWebView` is "generalized from `ArtifactWebView`", so it inherits this. The design still does not say what happens to widget state on reload, which is covered in the M12 justification.
* **SY-10 part of M14 ("ten gates" and "21 languages" lack a source): the counts are correct.** `swift/docs/SWIFT_LOCALIZATION.md` states 21 languages (line 37) and "Ten tests" for the parity gates (line 106). Only a link is missing.
* **N1 part ("662c30be has no link"): partly false.** `docs/ZCODE_COMPARISON.md` line 29 and the 4.3 provenance line already carry the commit with its message. The design should link there instead of restating bare text.

## Facts the doc gets wrong that no reviewer flagged

* **Doc line 28-29 ("the manifest parser records it as a one-line 'unsupported surface' diagnostic") is not true of the code.** `State/PluginManifestParser.swift` `collectUnsupportedNotes` handles `lspServers`, `outputStyles`, `channels`, `settings` and `.mcpb` only. There is no `ui` or `surfaces` reference in `PluginManifestParser.swift` or `PluginModels.swift`. A `ui.surfaces` entry is silently ignored today. This is folded into N6's edit. Goal 1 and file item 5 ("replace the unsupported-surface diagnostic") need the same correction.

## Prioritized Findings & AI Suggestions

### [Issue 1] [`egm-design / structural / style / zoom-out`] - Severity: Blocker (B1)
*   **Affected Scope:** [Lines 139-155](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L139-L155) (Alternatives Considered)
*   **Discrepancy Description:** Both alternatives are "Rejected by author" with no link. Alternative A mis-states the technical question (shared persistent store vs non-persistent store per view). Missing: extend `ArtifactWebView` with a bridge only, defer plugin surfaces or Gen UI, scheme-handler origin.
*   **Rationale:** Alternatives Rule. A reviewer cannot tell why the options were rejected.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Reading `ArtifactWebView.makeNSView`: every view already calls `.nonPersistent()`, so a "shared persistent store" is not a realistic baseline and Alternative A argues against a design nobody proposed. The real decision that was made implicitly is "new sibling host vs extend `ArtifactWebView`", which is also the subject of M2. Replacement text below contains new content that the author must confirm (marked).
*   ***Proposed Edit:***
    *   Location: `## Alternatives Considered`, replace Alternative A (lines 141-147) and append Alternatives C and D.
    *   Change:
        ```markdown
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
        ```

### [Issue 2] [`egm-design / structural / inventive-catalyst / comprehension / style`] - Severity: Blocker (B2)
*   **Affected Scope:** [Lines 36-62, 104-137, 157-168](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L36-L168) (Goals, Architecture, NFR; missing Risks section)
*   **Discrepancy Description:** No threat analysis or Risks and Mitigations section. `sendFollowUp` is a prompt-injection and self-prompting channel; page-initiated `invokeTool` has no rate limit, user gesture or origin label in the permission prompt; Gen UI has no manifest so its capability set is undefined; no sender-frame or iframe policy.
*   **Rationale:** The page is model- or plugin-authored script that can trigger tool calls and write text the model later reads.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid, and security relevant (injection, approvals, permissions). The doc line 96 says the follow-up "posts an attributed message into the composer", which does not say draft or auto-send, and "attributed" is undefined. Whether the injection channel exists depends on that choice, so the conservative default is draft-only. The same applies to `invokeTool`: the doc routes it through "the ordinary permission engine" (line 85), but a page can call it in a loop with no gesture, so a prompt flood is possible. Edits are deny/ask by default. Specific numbers are invented and need author confirmation.
*   ***Proposed Edit:***
    *   Location: insert a new `## Risks and Mitigations` section after `## Non-Functional Requirements` (after line 168).
    *   Change:
        ```markdown
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
        ```
        Also change line 163 and Goal 4 to match the draft-only rule.

### [Issue 3] [`egm-inventive-catalyst / codebase-details / comprehension / style`] - Severity: Blocker (B3)
*   **Affected Scope:** [Lines 42-46, 159-160, 177-178](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L42-L46) (Goals, NFR, Target Files item 1)
*   **Discrepancy Description:** "WKContentConfiguration-style CSP" is not a WebKit API and file loads cannot carry a CSP header. `nonPersistent()` returns a fresh store per call, so "partition per identity" and "frees the partition slot" are not WebKit concepts. Fix: serve page content through a `WKURLSchemeHandler` that injects CSP, and define identity.
*   **Rationale:** The CSP and isolation claims are the design's security boundary.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:* Security relevant (CSP, isolation).** What I could verify from readable files:
    *   No type named `WKContentConfiguration` appears anywhere under `swift/` (searched). The repo's existing mechanisms are `WKContentRuleList` (`ArtifactWebView.swift`, `ArtifactContentRuleList`; `AgentBrowserIsolation.swift`) and a `<meta http-equiv="Content-Security-Policy">` in `State/AppChatShareDocument.swift` line 65.
    *   `ArtifactWebView.makeNSView` sets `configuration.websiteDataStore = .nonPersistent()` and its doc comment says "nothing shared with any other web view", which is consistent with a fresh store per call. So "partition slot" has no meaning for a non-persistent store: there is nothing to allocate or free beyond releasing the view. This part of the reviewers' claim is supported by repo code.
    *   `loadFileURL(allowingReadAccessTo:)` and `loadHTMLString(baseURL: nil)` are what the host uses today (lines 170, 172).
    *   What I cannot verify from readable files: that WebKit has no CSP-injection API for file loads, that `WKURLSchemeHandler` is the right mechanism, and the exact semantics of `WKWebsiteDataStore(forIdentifier:)` (used at `AgentBrowserIsolation.swift` line 114, which suggests identifier-keyed persistent stores do exist, and are the only thing that could be called a "partition"). These need confirmation against Apple's WebKit documentation, so I am not asserting them.
    *   The sender-frame claims are likewise unverified here.
    The conservative rewording that is safe whether or not the scheme-handler claim holds: remove the name "WKContentConfiguration", say "CSP injected by the app, mechanism decided in implementation", and replace "partition" with "per-view non-persistent data store". The user should decide the mechanism.
*   ***Proposed Edit:*** (offered for the user's decision, not a PROPOSE FIX)
    *   Location: Goal 3 (lines 43-46), line 115, lines 159-161, 177-179, CUJ 1 line 86-87.
    *   Change: replace "`WKWebsiteDataStore` partition, derived `WKContentConfiguration`-style CSP" with "its own non-persistent `WKWebsiteDataStore` (one per live page) and a CSP derived from the manifest's resource declarations, enforced by [mechanism: user to choose, candidate `WKURLSchemeHandler` serving all page content]"; replace "frees the partition slot" with "releases the page's data store"; define "page identity" as plugin id + surface id (+ session id for inline cards).

### [Issue 4] [`egm-zoom-out / inventive-catalyst / design`] - Severity: Major (M1)
*   **Affected Scope:** [Lines 17-23, 47-54, 170-228](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L17-L228) (Overview, Goals, Plan)
*   **Discrepancy Description:** Three findings bundled with no phasing; Gen UI does not need manifests, a registry or plugin partitions.
*   **Rationale:** Smallest viable first step is hidden.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Gen UI renders agent-written pages from the chat artifact folder, which `ArtifactWebView` already loads (`ArtifactWebDocument.file`). The manifest model and live-page registry serve only plugin surfaces. Phasing is a human-reader need, not only a reviewer one.
*   ***Proposed Edit:***
    *   Location: after `## Goals`, add `## Phasing`.
    *   Change:
        ```markdown
        ## Phasing

        * **Phase 1: Gen UI.** Extend `ArtifactWebView` with the page bridge
          (widget state and draft-only follow-up). No manifest, no registry.
        * **Phase 2: plugin surfaces.** Manifest model, `PluginUiHost`, side
          pane. Starts only when a real plugin declares a surface.
        ```
        Target Files items 1, 5, 6, 8 move to phase 2. [Author to confirm.]

### [Issue 5] [`egm-zoom-out / inventive-catalyst / design / style`] - Severity: Major (M2)
*   **Affected Scope:** [Lines 134-137, 180-186](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L134-L137) (Architecture, item 2)
*   **Discrepancy Description:** Sibling `PluginPageWebView` risks forking the security-critical sandbox; "fewer capabilities, never more" is false.
*   **Rationale:** Two copies of the sandbox drift.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:* Security relevant (sandbox).** Valid. `ArtifactWebView` is a small struct (`document`, `networkAllowed`) whose Coordinator holds the navigation policy, `allowsFileNavigation`, and termination handling. Copying it duplicates all of that. The "never more" sentence is wrong because the bridge adds script message handlers that an artifact does not have, and the grant flow is the same, so capability parity is the best claim.
*   ***Proposed Edit:***
    *   Location: lines 134-137 and Target Files item 2.
    *   Change: replace the paragraph with: "The existing `ArtifactWebView` grants model (offline by default, per-content network grant baked into view identity, ephemeral storage) stays the security floor. A plugin page has no more network or file authority than an equivalently granted artifact. The bridge adds script-to-app methods, each validated against the manifest." Replace item 2 with: "`ArtifactWebView.swift` (Modify): add optional bridge and CSP parameters to the existing view instead of a sibling type." Remove `PluginPageWebView.swift` from the create list and from the diagram (line 113).

### [Issue 6] [`egm-zoom-out / inventive-catalyst / design`] - Severity: Major (M3)
*   **Affected Scope:** [Lines 193-198, 51-54, 130](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L193-L198) (item 4)
*   **Discrepancy Description:** Gen UI controller and tools duplicate the artifact write/register path; a model-supplied path is a weak control.
*   **Rationale:** Duplicated write path and path-traversal surface.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:* Security relevant (path handling, tool approval).** Valid in principle: the doc itself says path "sanitized to the chat's artifact folder", i.e. it accepts a model-supplied path. I did not trace the existing artifact registrar in this pass, so the claim that a reusable registrar exists is unverified; the edit names it as the author's check.
*   ***Proposed Edit:***
    *   Location: Target Files item 4.
    *   Change: "write-page tool takes content only; the app generates the filename inside the chat's artifact folder; reuse the existing artifact write and registration path with a render-inline flag [author to confirm the registrar symbol]. Ladder: write is medium risk, render is low only for pages with no tool authority (see Risks)."

### [Issue 7] [`egm-zoom-out / inventive-catalyst / design / comprehension / style`] - Severity: Major (M4)
*   **Affected Scope:** [Lines 41-46, 106-112, 164-165](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L41-L46) (Goals, NFR)
*   **Discrepancy Description:** Cap 64 borrowed from Electron; no memory budget next to local model weights; "page identity" undefined; page 65 behavior undefined.
*   **Rationale:** The cap is a memory control on a machine that also holds model weights (CLAUDE.local.md: 36 GB unified memory).
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:* Valid.** ZCODE_COMPARISON 4.4 shows 64 is ZCode's number for Electron guests. The doc says "matching ZCode" with no TurboSpark rationale. The numbers below are inventions for the author to confirm.
*   ***Proposed Edit:***
    *   Location: Goal 2 and NFR Performance.
    *   Change: "a provisional cap of 8 live pages (author to confirm; ZCode uses 64 on Electron, and TurboSpark shares unified memory with model weights). Opening past the cap closes the least recently visible page after teardown; a page being interacted with is never evicted. Page identity is plugin id + surface id + session id." Name the constant `PluginUiHost.maxLivePages`.

### [Issue 8] [`egm-inventive-catalyst / comprehension`] - Severity: Major (M5)
*   **Affected Scope:** [Lines 17-19, 56-59, 79-85](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L17-L85) (Overview, Non-Goals, CUJ 1)
*   **Discrepancy Description:** Content source and flow unspecified: `resourceUri` implies MCP-fetched HTML, line 58 says on-disk only; the page reference in the tool result and how the user opens it are undefined; manifest fields lack an example.
*   **Rationale:** Implementer cannot tell where the HTML comes from.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; a human reader hits the same gap. Facts about the manifest wire format (`_meta.ui.resourceUri`) are in ZCODE_COMPARISON 4.3, so the example can be grounded there, but the TurboSpark choice is the author's.
*   ***Proposed Edit:***
    *   Location: after Goals, in CUJ 1.
    *   Change: add "Content source: the plugin ships page HTML in its bundle; `resourceUri` is a bundle-relative path, never fetched over the network or from the MCP server at render time [author to confirm]. A tool result references a page with `_meta.ui.resourceUri` (as in ZCODE_COMPARISON 4.3); the transcript row shows an Open button that opens the declared placement." Add an example block:
        ```json
        { "ui": { "surfaces": [ { "id": "chart", "resourceUri": "ui/chart.html",
            "resources": ["ui/chart.js"], "capabilities": ["invokeTool", "widgetState"],
            "placement": "sidePane" } ] } }
        ```
        [Author to confirm field names.]

### [Issue 9] [`egm-zoom-out / design / style / comprehension`] - Severity: Major (M6)
*   **Affected Scope:** [Lines 64-73 vs 51-53, 77-87](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L64-L73) (Open Questions)
*   **Discrepancy Description:** Open Questions contradict the plan: option (b) kills the vendor-resource goal; both placements are built while the default is open.
*   **Rationale:** Goal 5 requires bundled charts; Q2 leans to none.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:* Valid.** Goal 5 says "vendor resources (charts) bundled", Q2 says "Lean (b) day one" (no vendor deps). These conflict as written. The doc already leans on (b), so the smallest resolution is to drop the vendor clause, which the author can reverse.
*   ***Proposed Edit:***
    *   Location: Goal 5 (lines 51-54) and Open Questions.
    *   Change: delete ", with vendor resources (charts) bundled and mapped inside the partition" from Goal 5; replace Q2 with "Resolved: option (b), Gen UI pages are self-contained HTML; a bundled charts helper is a follow-up. [Author to confirm.]" Resolve Q1 per surface type as the doc already suggests: side pane for tool-result pages, inline for Gen UI answers.

### [Issue 10] [`egm-design / style / comprehension / structural`] - Severity: Major (M7)
*   **Affected Scope:** [Lines 56-62, 134-137, 182-186](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L56-L62) (Non-Goals, item 2)
*   **Discrepancy Description:** The network grant scope (origins, fetch/XHR, CSP connect-src) is unstated; the non-goal "no JS outside the webview" is unmeasurable and contradicts the injected bridge.
*   **Rationale:** An under-specified network grant is a security hole.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:* Security relevant (network authority).** Valid. Verified in `ArtifactWebView`: a network grant today allows all http/https/ws/wss and reloads the page inline under an opaque origin (`isolatedForNetworkAccess`), with no per-origin scoping. The doc line 184-186 promises a manifest-declared network grant with no scope. Safest for v1 is no network grant at all.
*   ***Proposed Edit:***
    *   Location: Target Files item 2 (lines 183-186) and Non-Goals line 60-61.
    *   Change: delete "network only if the manifest declares it and the user grants it, reusing the artifact grant flow" and write "no network grant for plugin pages in v1; pages are offline." Replace Non-Goal 2 with "No script runtime of its own beyond the injected bridge; the bridge is the only app-provided script surface" and move the measurable rule (offline, deny navigation) to NFR Security.

### [Issue 11] [`egm-structural / zoom-out`] - Severity: Major (M8)
*   **Affected Scope:** [Lines 98-102](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L98-L102) (CUJ 3)
*   **Discrepancy Description:** CUJ 3 has Given and Then but no When.
*   **Rationale:** Template shape.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; line 100 puts the trigger inside the Given.
*   ***Proposed Edit:***
    *   Location: CUJ 3.
    *   Change: Given: "a plugin page is open and its manifest does not declare `invokeTool`." When: "the page calls `window.turbospark.invokeTool(...)`." Keep the existing Then.

### [Issue 12] [`egm-zoom-out / comprehension / design`] - Severity: Major (M9)
*   **Affected Scope:** [Lines 47-52, 96, 120-125, 163](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L47-L52) (Goals, bridge)
*   **Discrepancy Description:** Bridge broader than any CUJ (`displayMode` unused); widget state scope unclear vs "no cross-page messaging"; display modes and error names not enumerated.
*   **Rationale:** Surface area with no consumer.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:* Valid.** CUJs use `invokeTool`, widget state and `sendFollowUp`; none uses `displayMode`.
*   ***Proposed Edit:***
    *   Location: Goal 4 and bridge block.
    *   Change: "v1 bridge: `invokeTool`, `widgetState` (get/set), `sendFollowUp` (draft-only). `displayMode` is deferred until a CUJ needs it. Widget state is keyed by page identity, so no page reads another's state. Refusal errors: `undeclared_method`, `permission_denied`, `payload_too_large`, `rate_limited` [author to confirm names]."

### [Issue 13] [`egm-inventive-catalyst / design`] - Severity: Major (M10)
*   **Affected Scope:** [Lines 47-50, 149-155](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L47-L50) (Goals, Alternative B)
*   **Discrepancy Description:** Bespoke `window.turbospark` bridge vs the MCP Apps interchange claimed in Alternative B; no spec named.
*   **Rationale:** Interchange claim without a wire contract.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:*** The MCP Apps specification and any wire-compatibility with ZCode's `window.zcode` live outside readable files. ZCODE_COMPARISON 4.3 names the page APIs but not a spec revision or postMessage dialect, and no TurboSpark file defines one. The user must say whether the bridge should be a shim over MCP Apps JSON-RPC postMessage or a bespoke API, and which spec revision. Interim edit if bespoke: reword Alternative B to "HTML pages are the ZCode and plugin ecosystem shape" and drop "MCP Apps" as an interchange claim.
*   ***Proposed Edit:*** none until the user decides.

### [Issue 14] [`egm-zoom-out / inventive-catalyst / design / style / structural`] - Severity: Major (M11)
*   **Affected Scope:** [Lines 217-223, 239-243](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L217-L243) (item 9, Verification Plan)
*   **Discrepancy Description:** No automated tests for navigation or permission denial, isolation, iframes, bridge origin spoofing; mutation checks sit in the unit-test bullets.
*   **Rationale:** Security claims are verified only manually.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:* Security relevant.** Valid. `swift/TurboSparkApp/Tests/TurboSparkAppTests/ArtifactWebViewSecurityTests.swift` exists and has two tests (file-to-inline conversion, inline stays inline), so extension is natural. `allowsFileNavigation` is already a pure static function, a good model for testable policy functions.
*   ***Proposed Edit:***
    *   Location: item 9 and Verification Plan.
    *   Change: add "Extend `ArtifactWebViewSecurityTests` with pure-function tests for: navigation to http(s), file outside folder and subframes refused; bridge rejects messages whose frame is not the main frame; bridge refuses undeclared methods and oversize payloads." Move the two mutation checks into a Verification Plan bullet "Mutation checks", and add one for the frame check.

### [Issue 15] [`egm-inventive-catalyst / zoom-out`] - Severity: Major (M12)
*   **Affected Scope:** [Lines 98-102, 159-165](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L98-L102) (CUJ 3, NFR)
*   **Discrepancy Description:** Failure handling is passive: no web content process termination, hang or load-failure path.
*   **Rationale:** Pages will be killed under memory pressure.
*   ***AI Suggestion:*** CONTEST
*   ***Justification:* The reviewer misread the codebase for the termination case.** `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ArtifactWebView.swift`, `Coordinator.webViewWebContentProcessDidTerminate` and `shouldReloadAfterTermination`: reload once, then `ArtifactWebView.stoppedHTML`. Doc line 113 says `PluginPageWebView` is "generalized from `ArtifactWebView`", and with M2's fix it is the same view. Load-failure: the same file fails closed to `offlineUnavailableHTML` when the rule list is missing. What remains is that widget state is not restored after the reload, which is a minor, deferrable gap (reviewer S8 also suggested deferring). No fix is proposed. This is not a comprehension gap: the doc does inherit the behavior, but a one-sentence note ("inherits the one-reload then stopped-state handling") would help a human reader, so the author may add it at no cost.
*   ***Proposed Edit:*** none (optional one sentence in NFR Performance).

### [Issue 16] [`egm-zoom-out / inventive-catalyst`] - Severity: Major (M13)
*   **Affected Scope:** [Lines 213-216, 36-40](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L213-L216) (item 8)
*   **Discrepancy Description:** Side pane routed into `SplitChatPanes` (read-only chat panes) instead of the artifact panel; no first concrete plugin needs a surface; lifecycle under chat switch and ZP-23 undefined.
*   **Rationale:** Wrong host and no consumer.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:* Verified, the reviewer is right.** `Generation/SplitChatPanes.swift` header: `SplitChatPanesView` renders `model.splitPaneChats`, "live follow surfaces" that are "read-only"; each pane is bound to a chat id. A plugin page is neither a chat nor read-only. The existing page host is `Generation/ArtifactPanelView.swift`, the only user of `ArtifactWebView(`. I did not verify ZP-23 contents.
*   ***Proposed Edit:***
    *   Location: item 8.
    *   Change: "Route the side pane through the artifact panel (`ArtifactPanelView.swift`), not `SplitChatPanes.swift`, whose panes are read-only chat followers. [Author to confirm.] First consumer: a test plugin shipped under the app's fixtures. Lifecycle: page is torn down on pane close, chat switch and session end; widget state is cleared with it." Add a three-row table (close, chat switch, app quit).

### [Issue 17] [`egm-design / style / comprehension`] - Severity: Major (M14)
*   **Affected Scope:** [Lines 4-7, 31-35, 188-192, 195-198, 237-238](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L188-L238) (various)
*   **Discrepancy Description:** Missing numbers and sources: message and write size bounds, ladder position per bridge method, "ten gates" and "21 languages", ZCode revision for parity claims, no source path for the "Swift app has a webview host" claim.
*   **Rationale:** Unbounded values cannot be implemented or tested.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid in part. Counts are correct (verified false sub-claim, see above): `swift/docs/SWIFT_LOCALIZATION.md` lines 37 and 106. `docs/designs/zcode-parity/INDEX.md` pins ZCode `a60ba6a7` / v3.17.1 (audited 2026-10-31), so the revision can be stated. The webview host is `Generation/ArtifactWebView.swift`. Byte limits are author decisions.
*   ***Proposed Edit:***
    *   Location: lines 3-7, 21, 189-192, 195-198, 237-238.
    *   Change: add "ZCode revision `a60ba6a7` (v3.17.1), audited 2026-10-31 (INDEX.md)"; line 21 cite `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ArtifactWebView.swift`; link "ten gates" and "21 languages" to `swift/docs/SWIFT_LOCALIZATION.md`; bridge messages "at most [N] KB, TBD by author"; Gen UI page writes "at most [N] KB, TBD by author"; add a ladder column to the bridge table (invokeTool: asks per the tool's own ladder; widgetState: low; sendFollowUp: medium, draft-only).

### [Issue 18] [`egm-structural / design / zoom-out`] - Severity: Nit (N1)
*   **Affected Scope:** [Lines 3-8](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L3-L8)
*   **Discrepancy Description:** Commit "662c30be, 426 files" is bare text; no verification date; no adopted/changed/dropped parity table.
*   **Rationale:** Provenance.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Partly overtaken: `docs/ZCODE_COMPARISON.md` line 29 and the 4.3 provenance line already record the commit and its message. Link there and add the audit date.
*   ***Proposed Edit:***
    *   Location: line 7.
    *   Change: replace "(662c30be, 426 files)" with "([662c30be](../../ZCODE_COMPARISON.md), audited 2026-10-31)". Skip the parity table unless the author wants one.

### [Issue 19] [`egm-zoom-out / comprehension / structural`] - Severity: Nit (N2)
*   **Affected Scope:** [Lines 106-132](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L106-L132) (diagram)
*   **Discrepancy Description:** Upward "declares" arrow; `GenUiController` not linked to `PluginUiHost`.
*   **Rationale:** Diagram readability.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; lines 126-128 draw an upward arrow into the bridge and place the parser inside the same chain.
*   ***Proposed Edit:***
    *   Location: the text block.
    *   Change: redraw top to bottom: `PluginManifestParser -> PluginUiHost`, `GenUiController -> PluginUiHost` (second entry), then `PluginUiHost -> ArtifactWebView -> PageBridge`. Remove the `^ declares` lines. Adjust after M2 and M1.

### [Issue 20] [`egm-structural`] - Severity: Nit (N3)
*   **Affected Scope:** [Lines 64-73](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L64-L73)
*   **Discrepancy Description:** Open Questions precede the sections they depend on; Q1 has no owner.
*   **Rationale:** Order.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid and cheap. After M6 both questions are resolved, so the section may shrink to one line.
*   ***Proposed Edit:***
    *   Location: `## Open Questions`.
    *   Change: move after `## Alternatives Considered`; add "Owner: whit3rabbit" to any question still open.

### [Issue 21] [`egm-structural`] - Severity: Nit (N4)
*   **Affected Scope:** [Lines 225-228](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L225-L228) (item 10)
*   **Discrepancy Description:** Item 10 sub-bullets indented 3 spaces may not nest.
*   **Rationale:** Markdown rendering.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Lines 224-227 use 3 spaces where items 1-9 use 3 as well, but "10. " is one character wider, so nested content needs 4.
*   ***Proposed Edit:***
    *   Location: lines 225-228.
    *   Change: indent the three sub-bullets by 4 spaces.

### [Issue 22] [`egm-style / comprehension`] - Severity: Nit (N5)
*   **Affected Scope:** [Lines 17-19, 49](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L17-L19)
*   **Discrepancy Description:** Jargon undefined: "Gen UI as the agent-facing form", "Tweak-style controls", "Gen UI partition".
*   **Rationale:** Readability.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; a human reader needs this.
*   ***Proposed Edit:***
    *   Location: Overview and Goal 5.
    *   Change: "Gen UI: the agent writes a small interactive HTML page (for example sliders and charts) for one question and the chat renders it inline"; "Tweak-style controls" becomes "adjustable controls such as sliders"; replace "Gen UI partition" with "Gen UI page's own data store".

### [Issue 23] [`egm-style`] - Severity: Nit (N6)
*   **Affected Scope:** [Lines 21-23, 28-29](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L21-L29)
*   **Discrepancy Description:** "Without weakening its grants model" is an outcome claim; the diagnostic string is not quoted.
*   **Rationale:** Verifiability.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid, and verification found a larger error: no `ui.surfaces` diagnostic exists in `State/PluginManifestParser.swift` (see the section above); `collectUnsupportedNotes` does not mention `ui`. The doc states something false about current behavior.
*   ***Proposed Edit:***
    *   Location: lines 28-29, Goal 1 (line 38-40), Target Files item 5 (lines 199-204).
    *   Change: "A plugin can declare `ui.surfaces` today, but `PluginManifestParser` ignores the field with no diagnostic." Goal 1: "parse `ui.surfaces` into a typed model; a malformed entry produces a diagnostic." Item 5: delete "replace the unsupported-surface diagnostic with real parsing". Move "without weakening its grants model" to NFR Security as "no more authority than an equivalently granted artifact".

### [Issue 24] [`egm-style`] - Severity: Nit (N7)
*   **Affected Scope:** [Lines 143-144](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L143-L144)
*   **Discrepancy Description:** "Compromised page reads another's storage" lacks its condition; "exist for exactly this" speculates about ZCode.
*   **Rationale:** Unsupported claim.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; handled by the Alternative A rewrite in Issue 1. If that edit is not taken, apply: "Cons: with a shared store, one page could read another's cookies and localStorage" and delete the ZCode motive clause.
*   ***Proposed Edit:***
    *   Location: lines 143-144.
    *   Change: as stated above; no new text beyond Issue 1.

### [Issue 25] [`egm-style`] - Severity: Nit (N8)
*   **Affected Scope:** [Lines 49, 117, 160](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L49)
*   **Discrepancy Description:** Deny-by-default stated three times.
*   **Rationale:** Repetition.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Security rule: when deduplicating, keep the NFR statement and do not weaken it.
*   ***Proposed Edit:***
    *   Location: line 49 and line 117.
    *   Change: delete "Everything not declared is refused." in Goal 4 and the "everything denied except..." wording in the diagram; keep line 159-160 in NFR.

### [Issue 26] [`egm-style`] - Severity: Nit (N9)
*   **Affected Scope:** [Lines 49, 189](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L189)
*   **Discrepancy Description:** "Facade" and "bridge" are used interchangeably.
*   **Rationale:** Terminology.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; line 189 says "`window.turbospark` facade".
*   ***Proposed Edit:***
    *   Location: line 189.
    *   Change: replace "facade" with "bridge".

### [Issue 27] [`egm-style / design`] - Severity: Nit (N10)
*   **Affected Scope:** [Lines 166-168, 224-228](../../../designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md#L166-L168)
*   **Discrepancy Description:** "Like the other subsystems" is vague; the page-cap constant is not named or located.
*   **Rationale:** Specificity.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. `swift/docs/` has per-subsystem pages (for example `SWIFT_PLUGINS.md`, `SWIFT_GHOST_MODE.md`), so one can be named.
*   ***Proposed Edit:***
    *   Location: lines 167-168.
    *   Change: "the sandbox contract gets a section in `swift/docs/SWIFT_PLUGINS.md`, like `SWIFT_GHOST_MODE.md` does for its subsystem [author to confirm model page]"; name the constant `PluginUiHost.maxLivePages` (see Issue 7).

### [Issue 28] [`egm-comprehension`] - Severity: Nit (N11)
*   **Affected Scope:** whole file
*   **Discrepancy Description:** Undefined project terms: ZCode, MCP Apps, DESIGN.md, design-check, mutation checks, "permission engine", "ladder".
*   **Rationale:** Terms known only to insiders.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid for a human new to the repo, but the doc sits in a design set whose INDEX.md already defines ZCode. Smallest edit: one glossary line.
*   ***Proposed Edit:***
    *   Location: after the metadata block (line 11).
    *   Change: add "Terms: ZCode and the ZP set are defined in [INDEX.md](INDEX.md); DESIGN.md is the app UI rule set; `make design-check` is its static scan; a mutation check breaks the guard on purpose and confirms the test fails; the ladder is the tool risk ladder the permission engine uses [author to link the defining doc]."

## ESCALATE items for the user

1. **B3 (WebKit CSP and partition claims).** I could not verify WebKit API claims from readable files. Please confirm against Apple's WebKit documentation whether CSP can be injected on `loadFileURL` loads, whether `WKURLSchemeHandler` is the mechanism to use, and whether any "partition" concept applies beyond `WKWebsiteDataStore(forIdentifier:)`. Repo evidence only shows `WKContentRuleList`, a meta CSP in `AppChatShareDocument.swift`, and a per-view `.nonPersistent()` store.
2. **M10 (MCP Apps wire compatibility).** Decide whether `window.turbospark` is a shim over MCP Apps postMessage JSON-RPC (and which spec revision) or a bespoke API.

Note on `sendFollowUp`: the reviewers' injection claim holds as a design-level risk because the doc leaves auto-send versus draft undefined (line 96, "posts an attributed message into the composer"). It is not a WebKit API question and I verified it only against the doc text, so the proposed edit makes it draft-only by default.
