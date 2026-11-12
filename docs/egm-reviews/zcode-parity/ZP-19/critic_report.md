# Stage 7: Critic report, ZP-19 web image search

Target: full file (168 lines). One round only; convergence not assessed. Reviewers egm-design (DC, feature-spec-standard), egm-structural (ST, feature-spec-standard), egm-style (SY). The first style spawn failed with "Concurrent subagent limit reached" and was retried once successfully. All completed.

## Design (DC)

- DC-1 (Blocker) Lines 36-37, 79-96. Precedence "MCP first" does not mirror WebSearchExecutor, which is a hardcoded provider switch (Brave, SearXNG, Parallel, Exa, Tavily, auto fallback Exa then Parallel) and never calls user-configured MCP servers. An MCP backend is new capability needing a schema contract, discovery rule, integration point, winner rule and failure behavior. Fix: describe the MCP-client integration or drop MCP from day one.
- DC-2 (Blocker) Lines 40-42, 68-69, 91-93, 163. `image_fetch` and the "attachment URL intake" do not exist as an agent tool; the only matches are composer `@url:` code. It is an unspecified second new tool and the real attack surface. Fix: add it as a deliverable (schema, ladder rung above search, content-type and magic-byte checks, size cap, destination path, redirect and rebinding handling via HttpRequestExecutor) or name the reused tool and verify binary handling; say where bytes land for ZP-09.
- DC-3 (Major) Lines 95-96, 118. "Reuse sanitization wholesale" is wrong for image rows: sanitize is text-shaped. Image and thumbnail URLs are attacker strings. Fix: list image-specific sanitization (scheme allowlist, length caps, control chars, no data: URIs, domain filters on both hosts, private-host drop at search time, untrusted marking).
- DC-4 (Major) Lines 49, 120-121, 144-146. `SearchSettingsPaneView.swift` not found; no Brave key settings or Keychain wiring found (braveApiKey is only a parameter). Fix: locate the real storage or scope the settings and Keychain work as new.
- DC-5 (Major) Lines 122-123. 90 s timeout copied from a hosted service; text path default is 25 s; a 90 s call stalls a single-active-chat app. Default result cap 8 conflicts with text path defaults (5, max 20) and CUJ 1 (5). Fix: use 25 s, state cancellation, pick one count.
- DC-6 (Major) Lines 99-114. "Rejected by author" rationales with no link; Alt B "unreliable" is not a trade-off; missing alternatives (reuse Exa/Tavily/SearXNG, Brave-only day one, image mode on web_search). Fix: concrete reasons, add alternatives.
- DC-7 (Major) No Risks section. Missing: licensing of embedded images, prompt injection via titles, content safety (safesearch), privacy of query text sent to third parties, Brave quota and 429, hotlink 403 and expired URLs. Fix: add the section.
- DC-8 (Major) Lines 71-77. CUJ 2 undecided (listed or hidden); text tool behavior not cited. Fix: pick one and cite.
- DC-9 (Major) Lines 140-143. Ladder position hand-waved; websearch, webfetch, http_request are grouped in AppToolCatalog.swift; the file list omits ToolPresentation.swift, ToolCallNormalization.swift, AppHookToolNameAliases.swift, AppToolCatalog.swift, AppToolRegistry.swift, so a registered tool would not dispatch. Fix: name rung per tool and add the files.
- DC-10 (Major) Lines 149-164. Tests miss fetch risks (rebinding, redirect to private, non-image MIME), prompt-injection title test, SSRF mutation check, localization gate. Fix: add.
- DC-11 (Nit) Line 125. "21 languages" disagrees with repo lproj count and INDEX "ten gates". Fix: cite verified count.
- DC-12 (Nit) Lines 3-6, 22-23. "Parity target" with no defined schema; say whether ZCode's argument names are adopted.
- DC-13 (Nit) Lines 8-10. INDEX sizes this small; scope is closer to medium. Fix: re-size or cut scope.

## Structural (ST)

Heading hierarchy and Markdown are clean; findings are missing or non-conforming sections.

- ST-1 (Blocker) Whole document (near 116-125). No Risks & Mitigations section. Fix: add H2 with risk, impact, mitigation.
- ST-2 (Major) Lines 106-107, 114. Rejections attributed to "the author" with no link. Fix: state trade-offs.
- ST-3 (Major) Lines 99-114. Missing alternatives: web_search image mode, MCP-only, other image APIs. Fix: add with rationale.
- ST-4 (Major) Lines 39-41, 91-93, 129-155. `image_fetch` absent from architecture and file plan. Fix: state new or existing; add files, schema, ladder.
- ST-5 (Major) Lines 140-143. Ladder position not stated as INDEX line 121 requires, and not stated for image_fetch. Fix: name exact rung per tool.
- ST-6 (Major) Lines 53-59, 76-77. Day-one MCP scope left to "author decides" and conflicts with Goals; CUJ 2 Then clause untestable. Fix: pick scope and behavior.
- ST-7 (Major) Lines 1-10. Missing the Sources block INDEX 126-129 requires (ZCode `.mcp.json`, Swift search note, `docs/document-skills-plugin.md`). Fix: add.
- ST-8 (Nit) Line 125. 21 versus ten gates.
- ST-9 (Nit) Lines 157-164. Verification omits `make design-check`, localization gates, and the local-green caveat.
- ST-10 (Nit) Lines 144, 154-155. Aside packed into file label; mutation checks mis-nested.
- ST-11 (Nit) Lines 33-38, 82-85. Inconsistent terms for backends and tool name.

## Style (SY)

- SY-1 (Major) Lines 14-20, 105. Claims about ZCode office plugins delegating to image search and splitting search from fetch are uncited. Fix: cite or narrow.
- SY-2 (Major) Lines 24-25. "greps match nothing" unscoped. Fix: state paths and date.
- SY-3 (Major) Lines 28-29, 95-97. Claims about existing code with no file; "wholesale" overstates; SSRF ownership attributed inconsistently (intake versus WebSearchExecutor). Fix: name files and guards.
- SY-4 (Major) Lines 122-123. 90 s unsourced and tied to a hosted server this design does not use. Fix: cite or justify.
- SY-5 (Major) Lines 38-39, 53-59. Open question changes day-one scope and conflicts with Goals. Fix: state scope and trigger.
- SY-6 (Major) Lines 76-77, 137-138. Two behaviors in one Then clause. Fix: pick one.
- SY-7 (Major) Lines 33-34, 40, 125. image_fetch not in file list or ladder; language count unreconciled. Fix: as DC-2, DC-11.
- SY-8 (Nit) Line 142. "Low risk" is a label; give the tier name.
- SY-9 (Nit) Lines 106, 112, 114. Subjective rejection wording ("unreliable", "blows context", "hostile to automation"). Fix: concrete reasons.
- SY-10 (Nit) Lines 16-28, 38-39, 112. Informal wording; "zero-config", "companion", "twin" undefined. Fix: tighten.
- SY-11 (Nit) Lines 4-5. "Parity target" overstates since backend differs. Fix: "Reference implementation" or state scope.
- SY-12 (Nit) Lines 153-155. Mutation checks mis-leveled; "key never logged" lacks assertion method. Fix: move to Verification, specify log capture.
- SY-13 (Nit) Lines 118-121. Credential and per-origin rules uncited. Fix: link sources.
