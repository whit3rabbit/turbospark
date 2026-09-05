---
uuid: "7b3eeef2-d664-48a7-9b54-9b250768029e"
title: "TurboSparkApp: SKILL.state bounded agent state"
summary: "Opt-in per-project toggle, default off. Replaces append-only history with a JSON state document so a 50-step agent finishes instead of dying at step 30-35 on an HTTP 400"
tags: ["swift", "app", "agent"]
source: "docs/SKILL_STATE.md, swift/CLAUDE.md Gotcha 35"
created: "2026-09-05"
updated: "2026-09-05"
depends_on: ["a0863529-2ab6-4386-ab12-7465356d5bc8", "448c3480-812a-4052-a9f9-6dd35d430f69"]
---

## What is SKILL.state, and what does it actually prove?

Do not confuse this with the "Skills" marketplace subsystem (SKILL.md
files, `SkillManager`, see [[swift-app-skills-marketplace]]). SKILL.state
is a different feature, an implementation of arXiv 2608.26263: it replaces
append-only conversation history with a compact mutable JSON execution
state (`goal`, `files`, `facts`, `commands`, `next`) so an agent's prompt
stays roughly constant size regardless of step count.

Measured 2026-08-29/30 on three real local installs (gemma4, qwen38-27b,
gptoss-20b), one seed each plus a reproduced second seed on gptoss: all
three hit a 1.00 valid-patch rate across 50 steps with zero divergence,
against the paper's own 0.42 on a comparable open-weight model. The
append-only baseline on the identical event stream ran out of context and
stopped at step 30 to 35 of 50 with an HTTP 400.

Tokens: 3 to 5x fewer for MORE completed work (e.g. gptoss-20b: 29,931
tokens for 50 steps against 117,539 for 39). The saving compounds: on
gptoss, the append-only arm costs 1.5x more at 10 steps and 5.1x more at
40.

Shipped 2026-08-30, opt-in per project, OFF by default, so the
append-only path is byte-identical when unset.

## Don't

- Don't read the 1.00 valid-patch rate as refuting the paper's 0.42. The
  task here is far smaller (8 shelves, 14 items, a 5-key schema against
  their 500-shelf inventory). It shows compliance is reliable at THIS
  difficulty, not that the paper's failure modes don't exist at any scale.
- Don't grow the schema past the shipped 5 generic fields without running
  `AppSkillStateRealModelTests` against a real server first. Schema growth
  is exactly the axis the paper's own failure modes track.
- Don't parse tool calls before merging the state patch. The patch is JSON
  in the SAME reply that may carry a tool call. `applySkillStatePatch` runs
  first and strips the `<state_patch>` block, or `extractToolCalls`
  mistakes it for a bogus tool invocation.
- Don't partially apply an invalid patch. It's DROPPED whole and surfaces on
  `skillStateLastError`. A bad merge would silently corrupt every step after
  it. A dropped one costs only that step's bookkeeping.
- Don't author a per-project schema expecting better results. The shipped
  schema is deliberately generic (one hardcoded `AppSkillStateSchema`)
  rather than per-domain like the paper's own model, since a per-project
  one would be unusable until someone wrote it.
- Don't reach for steering vectors if the model favors one flat `facts`
  list over the structured fields. That's a decomposition-quality question
  the validator can't see, not a compliance failure. The cheap lever is
  `AppSkillStateSchema.fields`'s one-line documentation per field, exactly
  what the model reads.
- Don't expect this to help ordinary chat. It's for the app's project/agent
  mode doing many tool calls. A short task loses nothing on the append-only
  loop, and this saves tokens and wall-clock only, never memory footprint.
