# Stage 6: Comprehension report, ZP-90

Reviewer egm-comprehension completed. Read only persona, target, INDEX.md.

## Paraphrase
A decision log for 12 findings from a ZCode vs TurboSpark comparison. Nine are excluded as needing hosted infrastructure, remote workspaces or non-macOS support (1.4, 1.10, 3.6, 6.5, 6.6, 7.3, 7.4, 7.11, 7.12). 7.9 is split: log export belongs to `desktop-shell-hardening`, the hosted form half is excluded. 7.1 is adopted as ZP-22 (API keys only). 7.2 is permanently excluded by author decision 2026-11-11. A closing log adds unrelated decisions (ZP-23, ZP-15, ZP-16) and reopen conditions. The reviewer could paraphrase every section except 7.1, which contradicts the rest.

## Findings
CO-1 Major, L91-111 vs L20, 36, 162. 7.1 reads undecided while the rest says adopted. Fix: rewrite as "Adopted as ZP-22", point open questions to ZP-22.
CO-2 Major, L14-24. "Twelve... nine... one" accounts for 10; 1.10 and 3.6 rest on judgment, not the roadmap categories. Fix: state the 9+1+1+1 split and which rest on policy.
CO-3 Major, L3-4, 14, 47, 115, 122, 164. ZCode, Missing, Different, numbering, Z.AI/BigModel undefined. Fix: two-sentence preamble.
CO-4 Major, L160-170. Log mixes in 2.1/6.12/8.5; reopen list omits 1.10, 3.6, 7.2, 7.11; L75 conflicts. Fix: label extras, give every finding a reopen condition or "never".
CO-5 Major, L45-56. Tailscale-bound server, "single verified Tailscale IPv4", Bonjour loopback API unexplained and unlinked. Fix: name and link the server rule.
CO-6 Major, L91-105. "server-client knowledge", "local ring", composer picker, compaction, Keychain rules undefined; "tools local-only" ambiguous. Fix: define or drop (goes with CO-1).
CO-7 Nit, L113-119. 7.2 lacks Recommendation and the key-half note. Fix: add decision line and ZP-22 pointer.
CO-8 Nit, L128-134. ZP-22 follow-up and "usage ledger" not confirmable. Fix: link ZP-22 section.
CO-9 Nit, L136-145. Spec not linked; redaction source and owner unnamed. Fix: link, name owner.
CO-10 Nit, L147-152. Profile export, "excluded category", MCP undefined. Fix: link or describe.
CO-11 Nit, L78-84. Docker not discussed; permission ladder and sandbox designs unlinked. Fix: address Docker, link.
CO-12 Nit, L8. Status "Draft" vs final decisions. Fix: "Decided 2026-11-11".
