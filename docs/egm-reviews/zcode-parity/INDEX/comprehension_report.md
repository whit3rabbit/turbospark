# Stage 6: Comprehension report (INDEX.md, full file, document read alone)

Reviewer: egm-comprehension (completed). Read the target only, no GoldfishContext. Finding IDs CO-1 to CO-11.

## What this proposes

The document is an index for a program of 23 design drafts (ZP-01 to ZP-23, plus a consolidated ZP-90) that responds to the "Missing in Swift" findings in a comparison report between ZCode and this Swift app. The aim is to account for all 40 findings and implement those that suit a local-first, macOS-only, native-Swift inference app. It maps each finding to a draft, to an existing kiro spec, or to an excluded/decided record. It gives cross-draft dependencies and an 8-step build order, and it lists constraints every draft must follow (no Python/Node, localization gates, Keychain rules, permission ladder, verification gates).

## How the existing system works, per this document

The document describes the existing system only in fragments. It is a macOS app with a Metal GPU runtime in `crates/gpu`, written in Rust (edition 2021, MSRV 1.82) with a Swift UI layer. It has chats, a plugin system with marketplaces and skills, a permission engine (`AppToolPermissionEngine`, a "ladder"), Keychain-backed credentials, and settings.json. It also has localization parity gates, a design check (`make design-check`), and an "EGM design document template". The app currently does not run concurrent background chats (finding 2.1 was "Different"). All of this is taken from other files and not explained here.

## Findings

Major
- CO-1. Lines 63-74, 25-51: counts do not reconcile. ZP-90's "remaining 12" includes 7.1 (drafted as ZP-22), 7.9 (partly kiro) and 7.2 (excluded, yet "key half" in ZP-22). 4.3-4.5 are three findings in one row. A reader cannot confirm the total is 40. Fix: add a per-finding ledger (finding, disposition, owner) that sums to 40 and split ZP-90's list into excluded and adopted/moved.
- CO-2. Lines 49-51, 78-79 vs 92-105: ZP-23 is numbered after ZP-21 and ZP-22 though it is their prerequisite; step 6 says land ZP-23 "before ZP-02 goes live" while step 7 builds ZP-02. Unclear whether "before" is merge order or release order. Fix: state which, and note the numbering anomaly.
- CO-3. Lines 4, 134-135: "ZCode a60ba6a7 / v3.17.1" is never defined (competitor, reference product, upstream?). The named branch differs from the current branch (main). Fix: one sentence defining ZCode; give a commit SHA.
- CO-4. Lines 3-5, 57-61, 69, 118-124: key context lives only behind links (comparison report, kiro specs, roadmap, DESIGN.md, verification.md); "kiro spec" is undefined. Fix: define "kiro spec" inline and state the roadmap policy in one sentence.
- CO-5. Lines 97-104: sequencing rationale partial. "Some cross-cutting" and "persistence or protocol machinery" name no drafts; ZP-05 is step 4 but ZP-20 and ZP-18 (step 3) "would ship through" it; ZP-06 is in no step. Fix: add ZP-06, resolve the ZP-05 order, name the cross-cutting items.
- CO-6. Lines 89-90: "turn loop seam" is undefined here. Fix: define in one sentence or link the ZP-22 section.

Nit
- CO-7. Undefined project terms: Open Questions (21), EGM template (17), ladder position (121-122), the ten localization gates (116), no-Python/Node rule (46), Gen UI, UI plugins, page sandbox (34), off-peak and idle tasks (29), "Different" category (51), "Missing in Swift" (2), builtin marketplace seeding and CDN zip (31). Fix: short glossary or per-term link.
- CO-8. Lines 25-51: size labels (small, medium, large, largest) have no criterion. Fix: define the scale.
- CO-9. Line 123: "Local green is not release evidence" is idiomatic. Fix: say that passing local tests does not satisfy the release gates.
- CO-10. Lines 3-12, 126-135: audit date 2026-10-31 vs draft date 2026-11-11; not stated whether findings were re-verified against a newer ZCode. Fix: state it.
- CO-11. Line 50 vs 72-74: "7.2 key half" is unexplained and appears to contradict "7.2 excluded permanently". Fix: say that ZP-22 covers API-key entry only and cloud OAuth is excluded.
