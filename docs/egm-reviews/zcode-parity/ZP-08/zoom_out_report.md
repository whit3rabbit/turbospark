# Stage 5: Initial zoom out report (ZP-08 office document rendering)

Target: docs/designs/zcode-parity/ZP-08-office-document-rendering.md (full file, 205 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out (ZO)

Verdict: objective is clear, but the problem statement is stale and the plan proposes a large native renderer without a demonstrated gap.

ZO-1. Blocker. Lines 16-17, 24-26, 34-35, 173-176, 177-183. Problem statement is wrong about current behavior. `.opaque` in `ArtifactPanelView.swift` (lines 291-308) already renders a `QuickLookPreviewView` inline plus an "Open with default app" button. `FilePreviewView.swift` (lines 168-202) already routes attachments with empty extracted text to `QuickLookPreviewView` and keeps a "Model text" toggle. The gap is fidelity and navigation, not absence. Item 7 names an "opaque external-opener leaf" that does not exist. Fix: restate the problem against current QuickLook behavior and name what it fails at, with fixtures.

ZO-2. Blocker. Lines 129-136, 82-90, 35-43. QuickLook-first is rejected on unsupported grounds. A QLPreviewView does not write the source file; the mtime/size test can assert that. QuickLook already pages xlsx and pptx. "Rejected by author" has no data. Fix: evaluate current QuickLook output on representative agent-written files before any viewer code.

ZO-3. Major. Lines 34-47, 92-112, 153-194. YAGNI scope: three bespoke OOXML layout engines, a render model, a container and three views, while Non-Goals concede fidelity below QuickLook. Fix: keep QuickLook, add only what it lacks (read-only assertion, size handoff); if native, start with one format and phase the rest.

ZO-4. Major. Lines 98-110, 155-160, 169-172. Reuse claims unverified. `OfficeArchive` shells out to `/usr/bin/unzip`, returns bytes via `data(for:)`, with extraction budgets from `DocumentTextExtractor.Limits` and a cumulative `consumedBytes` counter. Rendering needs many parts (images, styles, media), so those limits will be exhausted or mis-sized, and each part spawns a subprocess. Item 6 "read-only handle API" is undefined and may be a no-op. Unclear whether `OfficeXMLParsers` can be extended or a separate parser family is needed. Fix: define the limits change for rendering, drop or define item 6, state the parser approach.

ZO-5. Major. Lines 45-46, 110-111, 155-194. Read-only invariant is mis-specified. Archive access already goes through a temp copy, so viewers never get a writable path. The read-only badge (line 165) has no stated user value, and the mutation test covers a path the design says must not exist. Fix: keep the mtime/size test, drop the architectural layer and the badge or justify it.

ZO-6. Major. Lines 138-146. 50 MB ceiling stated without relation to existing `DocumentTextExtractor.Limits`; cap rows and limits footer add UI states for unobserved failures; performance claims unquantified. Fix: reuse existing limits or explain the delta.

ZO-7. Major. Lines 55-56, 17-20, 40-43. VBA notice is likely dead code: docx/xlsx/pptx do not carry VBA (only xlsm/docm/pptm). Fix: say whether macro-enabled types are routed, else drop the notice and CUJ 2 clause.

ZO-8. Major. Lines 7-10, 41-44, 99. PDF "fourth viewer" adds no change: `PDFDocumentView` already exists (`ArtifactPanelView.swift` line 287). Fix: remove from deliverables or state no code change.

ZO-9. Major. Lines 58-64, 173-183. Open question is already answered by the code (extending is the established pattern), and the plan omits `FilePreviewView.swift`, the real routing site (`textOrQuickLookView`, `quickLookURL`, `hasRenderedAlternative`, lines 168-202). New kinds must preserve the Model-text toggle (CUJ 1). Fix: add the file and name the toggle constraint.

ZO-10. Major. Lines 181-184. Wrong path: `AppArtifactRenderKind` lives in `State/AppArtifact.swift` (enum at line 35), not `ArtifactPanelView+Kinds.swift`, which does not exist. The enum is Codable and String-backed, so adding cases changes the persisted form; no migration for persisted `.opaque` office entries; `displayLabel` and `forFileName` need updating. Fix: correct path, add persistence compatibility.

ZO-11. Nit. Lines 94-112, 153-194. Per-format parse detail (styles.xml run props, frozen panes, number formats) and fixture/mutation prescriptions are implementation altitude. Fix: move to a later implementation spec.

ZO-12. Nit. Line 14, INDEX line 81. ZP-08 is a prerequisite for ZP-09 review, but QuickLook already covers that need. Fix: soften the INDEX dependency.

ZO-13. Nit. Line 146 versus INDEX lines 115-116. "21 languages" versus "ten localization parity gates". Fix: reference the INDEX constraint.

ZO-14. Nit. Lines 4-6. ZCode renders in a web runtime with a CJK font runtime; the premise does not transfer to a native app with system fonts and QuickLook. CJK handling appears in the Problem and not in the plan. Fix: state what ZCode does that is worth matching, address or drop CJK.

Recommendation: do not proceed to implementation until the Problem section is rewritten around measured QuickLook gaps and the file list is corrected.

## Inventive catalyst (IC)

Files read: persona, ZP-08, INDEX.md, ZCODE_COMPARISON.md lines 1-237. Not verified against ZCode contract or Swift sources.

IC-1. Major. Lines 45-46, 110-112, 189-193. False compromise: a read-only handle does not stop the file changing during a parse, and ZP-09 (INDEX line 81) rewrites the same files. A mid-write docx reads as corrupt. The mtime/size test proves only no write by the viewer. Fix: parse a snapshot (APFS clone or hash-verified cache copy) and offer a "document changed, reload" affordance; cover snapshots under the vault's decrypted-attachment purge.

IC-2. Major. Lines 140-145, 114-116. Untrusted OOXML is parsed in-process, bounded only by caps and a 50 MB cliff; a crafted file can fault the app. Fix: parse in an XPC helper with a memory limit and per-sheet or per-slide on-demand parse. Costs: packaging and signing, serialization overhead. Not checked whether `OfficeArchive` has a streaming API.

IC-3. Major. Lines 129-136, 52-54, 153-193. QuickLook as "fallback for unsupported elements" (line 135) has no plan entry, yet themes, SmartArt, charts and footnotes are what QuickLook renders well. Fix: add a "Fidelity view" toggle using QLPreviewView on the snapshot, auto-offered when the limits footer reports unsupported elements, or delete line 135.

IC-4. Major. Lines 52, goals for pptx and xlsx styles. Skipping themes yields wrong or missing colors because most real fills are theme references (`schemeClr`, `theme="n"`). Fix: implement only the color-scheme map and major/minor font names; leave gradients and effects out.

IC-5. Major. Lines 141-145, Verification Plan. Failure handling is truncation and a size handoff only. Corrupt, encrypted, mid-document parse errors, legacy `.xls/.doc/.ppt`, strict OOXML, and silent cap hits are uncovered. Fix: define a degradation ladder (rendered, partial plus banner, existing "Model text" view, external opener) with failure fixtures.

IC-6. Nit. Lines 44, 183-184, 80. Missing plan entries: PDF "fourth viewer", the preview/source toggle CUJ 1 relies on, refresh path on rewrite, docx list/table/image fixtures. Fix: add rows or move to Non-Goals.

IC-7. Nit. Lines 146, 199 versus INDEX line 115. "21 languages" versus "ten localization parity gates". Fix: reference the gate by name.

No false compromise found in the rejection of Alternative A (WKWebView).
