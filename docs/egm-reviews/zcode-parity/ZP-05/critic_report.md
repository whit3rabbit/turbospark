# Stage 7: Critic Review (ZP-05), round 1

Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). All three completed. One round only; convergence not assessed. Standard: feature-spec-standard for design and structural. Note: DC found `agents/design.md` missing and used `agents/design_critic.md`.

## Design critic (DC)

DC-B1 (Blocker). Lines 116, 124. Both rejections read "Rejected by author:" with no linked issue or PR, and the reasons are thin. State technical reasons (A: bundle is a signed read-only .app, writes next to the plugin fail, app updates swap the target under ledger rows). Add alternatives for a first-party CDN and for manifest shape.

DC-B2 (Blocker). Lines 98-106, 129, 142-164. No zip extraction design. The existing installer handles directories and git, not archives; a search of State/Plugin*.swift found no zip handling. Missing: extractor component and target file, choice of ditto/unzip versus native reader, zip-slip, symlink escape, absolute paths, zip bomb and entry-count limits, download size cap, and tests for them.

DC-M1 (Major). Lines 62-65, 80-82, 129-130. Digest in the same row as the URL only catches corruption. Name the trust anchor (row origin plus HTTPS); say whether HTTP is refused, redirects followed, digest re-verified on reinstall; record signatures as a rejected alternative.

DC-M2 (Major). Lines 129-132. Not stated whether builtin hooks are trusted or still prompt. Seeded files in the user-writable profile may be tampered with after seeding; say whether the marker stores a content hash.

DC-M3 (Major). Lines 38-41, 89-107, 162-164. Seed lock undefined (location, stale lock recovery, behavior when held). Marker in the "ledger" conflicts with the existing v2 ledger (installed_plugins.json): schema decision missing. Suppression in settings.json has no rationale. Bundle update semantics (overwrite, skip, prompt, user-modified files) missing. Crash between copy and marker: stage, atomic move, write marker last.

DC-M4 (Major). Lines 99-100. `ranged_download` exists only in Rust repack code and tests, not Swift reusable. Define a URLSession download to temp file, streaming hash, max size, timeout, cancellation. Partial-download refusal has no defined behavior.

DC-M5 (Major). Lines 17-19, 135-137, 146-153. `builtin` case has no payload; `MarketplaceSource` is Codable so persisted decoding and downgrade are affected. `builtin` is a reserved name in SWIFT_PLUGINS.md lines 51 and 118; reconcile with `turbospark-plugins-builtin`. Can the user remove or edit the row; hide it when empty?

DC-M6 (Major). Lines 55-65. CDN manifest shape drives the data model and is already assumed. Resolve it or move it to Alternatives. State how ZCode's merged official row differs.

DC-N1 (Nit). Whole file. No success metrics or limits (seed time, bundle size, max zip size).
DC-N2 (Nit). Lines 90-91, 104. Say whether seeding runs off the main actor and whether it blocks first plugin load.
DC-N3 (Nit). Lines 154-177. No mention of localization parity gates or `make design-check`; name and localize the digest mismatch error; add a mutation check for extractor limits.

## Structural critic (ST)

No Blockers.

ST-M1 (Major). Whole document; lines 127-136. No Risks and Mitigations section. Failure modes (partial download, corrupt resource, stale lock, ledger/settings divergence), scale limits, transport trust missing.
ST-M2 (Major). Lines 116, 124. "Rejected by author" attribution; line 116 "delete a bundle path? no" is a fragment.
ST-M3 (Major). Lines 109-125. Only two alternatives, both about seeding. None for the CDN half; manifest shape sits in Open Questions; signed manifests, reuse of url/github source, and no seeding are unrecorded.
ST-M4 (Major). Lines 87-107 versus 146-149, 160-164. Architecture mentions only `cdnZip` but file 2 adds `builtin`; diagram omits suppression read and restore path; mutation check names a concurrent-seed test missing from the test list.
ST-M5 (Major). Lines 55-65. Manifest shape question is effectively decided; mark it resolved or make the plan conditional.
ST-N1 (Nit). Lines 3-7, 52, 58-59, 28. Links point at mutable main; INDEX pins a60ba6a7 / v3.17.1. Finding 4.9, ZP-18, ZP-20 unlinked.
ST-N2 (Nit). Lines 9-11, 99. Thin user motivation; "the Swift side" and `crates/repack ranged_download` lack introduction or path.
ST-N3 (Nit). Lines 142-179. Field labels inconsistent (item 2 "Proposed cases"; item 5 unlabeled "Mutation checks").
ST-N4 (Nit). INDEX consistency. No Dependencies line (ZP-20 and ZP-18 consume this).

## Style critic (SY)

No Blockers.

SY-M1 (Major). Lines 26-30. "Consequences" overstates: there is no builtin to uninstall today. Reword as a requirement; cite swift/docs/SWIFT_PLUGINS.md at first claim.
SY-M2 (Major). Lines 129-132, 84, 106. "Seeding never runs code" is absolute with no evidence; "existing plugin trust model" and "existing atomic installer" lack source links.
SY-M3 (Major). Lines 45-46, 80, 129-130. Behavior for a missing or malformed sha256 is unstated; "named error" never named.
SY-M4 (Major). Lines 55-65, 71. Open question vs plan assumption; CUJ 1 hard-codes "guide skills", mark illustrative.
SY-M5 (Major). Lines 114, 118-125. Subjective "muddy", "heavy and brittle"; give concrete costs.
SY-M6 (Major). Lines 133-137. Empty official segment is an undecided UI choice; no word on migration for existing profiles.
SY-N1 (Nit). Lines 3-7. No commit pin.
SY-N2 (Nit). Lines 18-22. Vague Overview, repeats Non-Goals.
SY-N3 (Nit). Lines 19-20. "gains the same capability" imprecise.
SY-N4 (Nit). Lines 98-100. Rust crate in a Swift path; "bounds safety" undefined.
SY-N5 (Nit). Lines 104-107, 96, 133. Suppression storage per-profile or global unstated.
SY-N6 (Nit). Lines 147-149, 42. Redundant "digest verification data"; singular "a new source kind" versus two kinds.
SY-N7 (Nit). Lines 163-164. Mutation checks not in Verification Plan; partial-download refusal has no matching Goal.
SY-N8 (Nit). Lines 171-178. Manual step run-on; release gate from .claude/docs/verification.md not named.
SY-N9 (Nit). Lines 1-2. Title "CDN zip source" versus `cdnZip`.
SY-N10 (Nit). Lines 51-52. Long parenthetical; "Partial" needs link.
SY-N11 (Nit). Lines 25, 58-60, 71. ZP-18/ZP-20 unlinked; use ZP-20 name once.
