**What this proposes**
Let plugins declare interactive web pages (`ui.surfaces` in the manifest) that render either as an inline card in the conversation or in a session side pane. Each page runs in a locked-down webview with its own non-persistent storage and a CSP derived from the manifest's declared resources. The page talks to the app only through a `window.turbospark` bridge (invoke declared MCP tools, send a follow-up message, get/set widget state, request a display mode). Everything undeclared is refused, and at most 64 pages are live at once. Gen UI is the agent-facing use of the same host: the agent writes an HTML page and an inline card renders it.

**How the existing system works, per this document**
`ArtifactWebView` is an existing webview that renders model-written HTML artifacts: offline by default, with a per-content network grant baked into the view's identity, and ephemeral storage. The manifest parser currently records `ui.surfaces` only as an "unsupported surface" diagnostic, and plugin MCP results show as ordinary tool rows. Tool calls go through a "permission engine" and a tool registry with "ladder" risk entries. The side pane would sit beside "existing read-only chat panes" in `SplitChatPanes.swift`. The document does not describe how any of these work.

**Findings**

CO-1. Major. Scope: Overview and Architecture, "ArtifactWebView grants model". The document says the grants model is "the security floor" and that a plugin page gets "fewer capabilities than an artifact with a granted network access, never more". It never lists what those grants are or what an artifact can do. The key claim cannot be checked, and "fewer" is not defined. Plugin pages can also be network-granted (file 2), which sits awkwardly with "fewer".

CO-2. Major. Scope: Page identity (Goals, Architecture "identity -> live page", Alternative A). "Page identity" drives partitions, the registry and the cap, but it is never defined. It could be plugin, surface, resourceUri, instance or conversation. It is unclear whether two open instances of the same surface share a partition or count twice against the cap. The cap of 64 also has no stated behavior when it is reached: refuse, evict, or queue.

CO-3. Major. Scope: Goals (CSP bullet) and Architecture. "WKContentConfiguration-style CSP" is not a known WebKit term (WKWebViewConfiguration and WKContentRuleList exist). It is unclear how a CSP is derived, or whether it is enforced via a content rule list or an injected header or meta tag. "Scriptable only from plugin-declared resources" is vague, and so is "Node-free", which appears to carry over from ZCode's Electron context.

CO-4. Major. Scope: `ui.surfaces` model `{resourceUri, resources[], capabilities[], placement?}`. No example manifest and no definition of each field. It is not stated what a resourceUri points to (a bundle path or an MCP resource), what a resource declaration contains, or what the capability vocabulary is. It is also unclear how capabilities map to bridge methods versus the "declared MCP tools". The reader cannot picture the manifest.

CO-5. Major. Scope: CUJ 1, "its MCP tool result carries the page reference and the user opens it". It is unspecified what the page reference is, how it appears in the tool result, and how the user opens it (a button, or automatic). The flow from tool result to side pane is a gap.

CO-6. Major. Scope: Gen UI. "Tweak-style controls" is undefined (a ZCode term?). "Gen UI partition" and "Gen UI identity" are used as if distinct from plugin identities, but never explained. The "Gen UI tool" in CUJ 2 versus the "write-page tool" and "render tool" in the implementation plan make it unclear whether there is one tool or two. The Open Questions treat "vendor resources (charts) bundled and mapped inside the partition" as a Goal, then lean toward no vendor bundle, an internal tension. How the Gen UI page's bridge capabilities are declared (there is no plugin manifest) is also not explained.

CO-7. Major. Scope: Bridge semantics. Unexplained: "attributed message into the composer" (attributed to what, and does it auto-send or only populate the composer), "widget state ... session-scoped store" (where, and how it relates to the CUJ 2 claim of "bridged"), and "display-mode change requests" (which modes exist). Non-Goals say "No cross-page messaging", but widget state is shared per session, so its scope is ambiguous. The "named error" for refusals is not enumerated.

CO-8. Major. Scope: Non-Goals versus Technical Plan. "No JavaScript execution outside the page's own webview context" is reconciled poorly with the PageBridge, which is a script message handler in the app. It also sits against "No remote page loading from arbitrary URLs" if a manifest can declare network access with a user grant (file 2, final Manual check). Which network origins can be declared is not stated.

CO-9. Major (Nit if documented elsewhere). Scope: "ladder entries (write is medium risk, render is low)", "ordinary permission engine", "tool registry". These project-specific systems are not defined or linked. Whether the user sees a prompt for bridge-initiated tool calls is not stated.

CO-10. Nit. Scope: Undefined project-specific references: "ZCode" and its feature branch (662c30be, 426 files) beyond links, "MCP Apps", "DESIGN.md", "design-check" and "design-check-full", "ten parity gates", the "21 languages" catalog, "swift/docs/ contract page", and "the Swift app". "ZCODE_COMPARISON.md findings 4.3 to 4.5" is not summarized.

CO-11. Nit. Scope: Open Questions. They state "The author decides" and a suggested default, while the Goals and CUJ 1 already assume side pane placement. The default for a surface without placement is unresolved but implemented in `PluginUiHost` routing. Alternatives A and B reasons are "Rejected by author" with a restatement of the con, which is thin.

CO-12. Nit. Scope: Implementation items 3 and 9. "Size bounds on messages" and "bounded size" for the Gen UI write have no values. "Mutation checks" is not explained. The "page-cap constant" in docs implies a named constant never given a name.

CO-13. Nit. Scope: Architecture diagram. The arrow direction and the placement of "PluginManifestParser" and "GenUiController" under PageBridge, labeled "declares" with an up arrow, make the relationships hard to read. GenUiController does not connect to PluginUiHost in the drawing, though the text says it does.
