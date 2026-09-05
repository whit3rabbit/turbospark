---
uuid: "1a2b3c4d-5e6f-4a1b-8c2d-3e4f5a6b7c8d"
title: "Swift binding: Catalog and SessionInfo API gotchas"
summary: "Read maxContext/expertCacheSlots off session.info, never off OpenOptions. liveTokenCount is not a throughput measurement. Build the reasoning picker from info.reasoningEfforts, not from allCases"
tags: ["swift", "binding"]
depends_on: ["50158806-656a-4a09-8bc1-39dc3801244b"]
source: "swift/CLAUDE.md"
created: "2026-09-05"
updated: "2026-09-05"
---

## What should I know before reading Catalog/SessionInfo/streaming fields?

A few API surfaces here answer a different question than their name
suggests. Each has already cost a real bug.

## Don't

- Don't drive an install progress bar from the last progress event alone.
  The install byte callback fires concurrently from several download
  threads, so progress can go BACKWARDS. Track a local running maximum
  instead. Install also runs on a DEDICATED `Thread` (blocks for tens of
  minutes), and a cancelled or failed install restarts from zero.
- Don't add `CodingKeys` to `CatalogEntry` or `InstalledModel` in
  `Catalog.swift`. These two decode snake_case deliberately, since they're
  the engine's own on-disk formats (`models.json`,
  `~/.turbospark/installed.json`) passed through unchanged, proving a
  GUI's rows and the CLI's rows are the same data. Everything else in this
  binding is camelCase, and `decode` does NOT set `keyDecodingStrategy`
  anywhere, on purpose: a spelling drift becomes a test failure instead of
  a silent nil.
- Don't read `maxContext` or `expertCacheSlots` off the `OpenOptions` you
  passed in. Read them off `session.info`, which holds what was RESOLVED,
  not what was asked for. Under automatic sizing, nothing was asked for.
  Same rule for speculation: `info.speculation.block != nil` IS the "is it
  on" test (there's no second flag to disagree with it), and non-nil says
  nothing about the NEXT turn, since acceptance is exact only at
  temperature 0.
- Don't quote `liveTokenCount` (the HUD counter) in a benchmark. It counts
  non-empty CONTENT events, not tokens: special tokens decode to empty
  strings, the detokenizer withholds partial UTF-8, and reasoning is a
  separate event. It undercounts on any turn with framing or thinking. Use
  `GenerationResult.newTokens`/`tokensPerSecond` instead.
- Don't add an emptiness guard to a future Swift-side event-stream parser
  by copying `streamCallback`'s `len == 0` drop. That drop is safe there
  only because the Rust-side state machine already ran. On Harmony, every
  frame token arrives as empty text, so a naive copy reads a whole turn as
  one run of content.
- Don't feed `reasoning` back into conversation history. Harmony and Qwen
  both drop prior-turn thinking, so replaying it sends text the model was
  never trained to read. `GenerationResult.content` is the assistant turn,
  `reasoning` is display-only.
- Don't build a reasoning-level picker from `Reasoning.allCases` or from
  the model family. The accepted set is a property of the checkpoint's own
  chat template (Qwen 3.8 refuses `.high`, gpt-oss has no `.xhigh`), and
  offering `allCases` shipped a menu entry that failed the turn until
  2026-08-31. Read it from `info.reasoningEfforts`, derived by actually
  rendering five prompts at open. A `.toggleOnly` checkpoint reports
  exactly two levels with the on-level at `.low` BY POSITION, so print
  "On", never the level's name.
