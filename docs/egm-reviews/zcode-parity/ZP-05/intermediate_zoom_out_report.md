# Stage 8: Intermediate Zoom Out Review (ZP-05)

Target unedited since Stage 5. No Blockers from this reviewer. Context sufficient. The root problem (bundled plugins need a delivery path, uninstall must stick) is real; the cdnZip half has no consumer and seeding duplicates existing install and state machinery.

## Stage 5 findings still open

All Stage 5 findings remain open (nothing was edited): ZO-1 (cdnZip has no consumer), ZO-2/IC-1 (Rust downloader reuse claim, no extraction bounds), ZO-3 (route seeding through the directory-source install path; persisted `builtin` kind and reserved-name collision), ZO-4/IC-3/IC-4 (lock scope, two stores, version-bump policy), IC-2 (digest in same row only catches corruption), IC-5, ZO-5, ZO-6, ZO-7/IC-6, IC-7.

## Scope recommendations on Comprehension and Critic asks

Accept (within altitude): CO-1, CO-2, CO-3, CO-5, CO-7 (define ledger, marker, suppression, version policy, how builtin row reaches the seeder, per-profile suppression); DC-M1 and DC-M2 (one sentence each on trust anchor and whether builtin hooks prompt); DC-B1 but downgrade to Major (replace "Rejected by author" with one technical reason each; a first-party CDN is already a Non-Goal).

Reject (adds scope or is misdirected):
- DC-B2 as a Blocker: designs an extractor for a feature that should be deferred. If cdnZip stays, one line requiring extraction bounds and traversal rejection is enough.
- DC-M4 detail, CO-6 "named error never named": implementation detail.
- ST-M1 (new Risks section) and ST-M3 (CDN, signed manifest, no-seeding alternatives): fold two or three failure modes into the NFR section only if cdnZip survives.
- DC-N1, DC-N2, DC-N3, SY-N8, ST-N3, ST-N4, SY-N1/ST-N1, SY-N2 to SY-N11, CO-8, CO-9, CO-10, SY-M5: polish, leave to a later style pass.
- SY-M3 only if cdnZip stays; DC-M5/SY-M6 keep only persisted-kind and name-collision parts.

## Net recommendation

Highest-value single edit: split cdnZip out (or name its consumer) and route builtin seeding through the existing directory-source install path with one ledger record carrying seeded, suppressed and version. That resolves ZO-1 to ZO-4, IC-1 to IC-5, CO-4, CO-6, DC-B2, DC-M4 and DC-M6 together without adding sections.
