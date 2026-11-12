# Stage 7: Critic report (ZP-08 office document rendering)

Target: full file (lines 1-205). One review round, no remediation. Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). Standard: feature-spec-standard for design and structural. All three completed. Limits: design, structural and style reviewers read ZCODE_COMPARISON.md lines 1-237 only, so ZCode claims were not checked against finding 5.7.

## Design (DC)

DC-1. Blocker. Lines 114-115, 140-141, 169-172. Security reuse claim does not match `OfficeArchive`. It shells out to `/usr/bin/unzip -p` per entry, copies the file to a temp `doc.zip`, and enforces `DocumentTextExtractor.Limits` with a cumulative `consumedBytes`. Rendering needs images, all sheets, styles, sharedStrings and every slide, which exhausts or forces raising those limits; the draft states neither. One subprocess per part is a latency risk for large decks. "Path containment" does not apply to `-p` streaming; the real gap is attacker-controlled `.rels` targets (`../media/x.png`, external links), which have no rule. Fix: add a Render Limits subsection with numbers (per entry, total, image bytes, entry count), decide unzip versus in-process reader, add rels-target validation and a ban on external references.

DC-2. Blocker. Lines 45-46, 110-111, 169-172, 193. The read-only invariant is assertion only and does not match the code: `OfficeArchive` takes a URL and copies the file, there is no handle API. The mtime/size test passes if the viewer writes elsewhere; the temp copy is itself a write. Extension routing is undefined for `.docm/.xlsm/.pptm` and legacy `.doc/.xls/.ppt`; external-link refresh and OLE objects are unmentioned. Fix: define the invariant as no writes to the source path and no network or process spawn except the sanctioned unzip; specify extension routing.

DC-3. Major. Lines 126-127, 135-136. Both rejections are "Rejected by author" with no linked decision; QuickLook claims are unevidenced. Missing alternatives: render to PDF and reuse PDFKit, thumbnail-only QuickLook, better text view only. Fix: concrete trade-offs, add alternatives, cite ZP-06.

DC-4. Major. Line 181, item 9. `AppArtifactRenderKind` is in `State/AppArtifact.swift` (line 35), not `ArtifactPanelView+Kinds.swift` (no such file). The enum is Codable String-backed and `forFileName` gates artifact registration; adding cases breaks decoding by older builds. `displayLabel` (line 239) and `ArtifactPanelView.swift` line 291 switch on the kind and are not listed. Fix: correct the path, list every switch site, specify unknown-raw-value fallback to `.opaque`.

DC-5. Major. Lines 143-145. No cost model: 50 MB of OOXML expands 10 to 100 times; a sheet can declare 1M rows; "first N" has no N; no parse timeout or cancellation on preview close. Fix: concrete caps, timeout, cancellation.

DC-6. Major. Lines 34-43 versus 52-56. Theme exclusion breaks "cell styles" and text color in the common case. No fidelity matrix, corpus or success metric; merged cells, hidden sheets/rows, RTL and CJK are not mentioned; the CJK claim from the Problem is never planned. Fix: supported/degraded/placeholder matrix, small real-world fixture corpus, fidelity floor.

DC-7. Major. Lines 32, 44, 99. PDF "fourth viewer" has no spec, file or test. Fix: scope out or specify.

DC-8. Major. Lines 22-30, 138-149. Unsupported value claim (review loop depends on git pane and rewind draft, no links); ZP-08 justification is circular with ZP-09 (INDEX line 81); no failure-mode list (corrupt, encrypted, file changing or half-written while viewed); no accessibility plan for a custom grid and slide navigator; "21 languages" versus INDEX "ten localization parity gates". Fix: state standalone value, add failure modes and VoiceOver note, reconcile the count.

DC-9. Nit. Lines 58-64. The open question is already answered. Fix: decide it.

DC-10. Nit. Lines 74-77. VBA notice cannot trigger for `.xlsx` (VBA lives in `.xlsm`). Fix: clarify routed extensions.

DC-11. Nit. Lines 187-193. Mutation checks nested inside the Create entry; inherited zip-bomb tests unconfirmed; no view or snapshot test for the three views. Fix: confirm tests exist, add view tests.

DC-12. Nit. Lines 3-10. Cites only CONTRACT.md; ZCODE_COMPARISON line 461 also names DocxView.tsx; no decision-owner field. Fix: cite both.

## Structural (ST)

ST-1. Blocker. Whole document. No Risks & Mitigations section, required by the standard. NFR has one Security and one Performance bullet, neither framed as risk plus mitigation. Missing: malformed or encrypted files, memory blowup, divergence from Office with no user signal, read-only failure in practice, external relationship targets. Fix: add the section.

ST-2. Major. Lines 126-127, 135-136. Alternatives Rule violated: "Rejected by author" without link; "keeps the trust boundary simple" is not a trade-off; ZP-06 unlinked. Fix: link or replace with concrete trade-offs.

ST-3. Major. Lines 118-136. Only two alternatives; viable ones missing (PDF conversion, third-party OOXML library, text-only plus better handoff, phased delivery). QuickLook is named the fallback (line 136) yet appears in no goal, diagram, file or test. Fix: add alternatives and either plan or drop the fallback.

ST-4. Major. Lines 58-64. Open Question already answered by items 7-9; no decision criteria or owner. Fix: resolve into the plan.

ST-5. Major. Lines 70-73, 55-56, 82-83, 144-145, 44, 164, 185-191. Goal/plan traceability gaps: preview/source toggle not in Goals or file list; VBA notice is a Non-Goal and a CUJ criterion; 50 MB handoff only in NFR; PDF has no file; no fixtures for comments, frozen headers, number formats; docx jump bar not in Goals. Fix: align Goals, CUJs and file list.

ST-6. Major. Lines 146, 195-202. "21 languages" versus INDEX "ten localization parity gates"; `make design-check` absent from Verification Plan; no reference to `.claude/docs/verification.md`; no view-level tests. Fix: reference the gates by name, add design-check.

ST-7. Major. Lines 51, 28-29, 124. Missing cross-draft dependency statements: what ZP-09 consumes from this draft; "git pane and rewind draft" unlinked (INDEX lines 57-59 give kiro spec paths); ZP-06 unlinked. Fix: add links and a forward reference.

ST-8. Major. Whole plan. No phasing for a draft sized "large" (INDEX line 36): three renderers in one plan, no delivery order, per-viewer gate or rollback; no data flow for threading, lazy load, cache lifetime or cancellation. Fix: phase per format.

ST-9. Nit. Lines 153-194. Items 3-5 lack Type/Scope/Changes; item 9 has a garbled parenthetical; item 10 indentation differs; items 7-9 have no test entry. Fix: normalize.

ST-10. Nit. Lines 22, 92. Headings differ from standard names. Fix: "Problem Statement", "Proposed Architecture".

ST-11. Nit. Lines 1-10. No link to INDEX.md or to existing Swift sources; CONTRACT.md link not pinned to a commit (INDEX line 133). Fix: add links, pin commit.

ST-12. Nit. Lines 94-112. ASCII diagram misaligned; unclear whether the new passes sit inside or beside the parser layer. Fix: realign.

## Style (SY)

SY-1. Major. Lines 4-5, 24-28. ZCode contract and "CJK font runtime" claims have no quoted source beyond the CONTRACT.md link; whether TurboSpark must handle CJK is undefined. Fix: cite sections or drop; add a goal or Non-Goal.

SY-2. Major. Lines 28-29. Causal claim about git pane and rewind draft unsupported and unnamed. Fix: name the spec or reword to "users must leave the app".

SY-3. Major. Lines 34-36, 60, 181-184. `FilePreviewView` is named in line 60 but missing from the file list; stray "basename" shorthand. Fix: add the file, drop "basename".

SY-4. Major. Lines 46, 111, 190-193. Read-only enforcement overstated; mtime/size test misses a size-preserving write. Fix: state the limit or use a content hash.

SY-5. Major. Line 192. Zip-bomb coverage claimed by reference to tests not shown to exist or to cover the new API. Fix: name the test file or add a test.

SY-6. Major. Lines 142-145. "bounded element counts", "first N", "clean handoff" and the 50 MB ceiling are unmeasured. Fix: N per format; replace "clean" with observable behavior.

SY-7. Major. Lines 38-43, 149. "basic shape rendering", "where sensible", "cell styles" undefined; theme exclusion conflicts. Fix: define the shape set and theme fallback.

SY-8. Major. Lines 127, 136. "keep the trust boundary simple", "quality varies by macOS version", "no navigation API surface" uncited. Fix: name versions and API, or drop.

SY-9. Nit. Lines 16-17. "preview surfaces" undefined at first use.

SY-10. Nit. Lines 29, 126. "weakens", "easier" subjective.

SY-11. Nit. Lines 58, 61. "Lean extending both" is not a question.

SY-12. Nit. Lines 52-56. "Documented limit" has no home; charts get a placeholder, footnotes nothing. Fix: point to the footer spec.

SY-13. Nit. Lines 100-108 versus 169. `OfficeXMLParsers` "extended" but not in the modify list.

SY-14. Nit. Line 148. "where sensible" hedging; name the setting or drop.

SY-15. Nit. Lines 185, 192. Item 10 indentation; "mutation checks" is jargon (say "negative tests").

SY-16. Nit. Lines 146, 199. "21 languages" versus INDEX.

SY-17. Nit. Lines 77-90. CUJ 3 "documented limits list" lacks assertions; CUJ 2 frozen panes mapping unspecified.

SY-18. Nit. Lines 12-31. Overview and Problem repeat "users leave the app".

SY-19. Nit. Dependency. Add a one-line forward link to ZP-09.

No emojis or em dashes found in the draft.
