---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e05"
title: "Getting a model"
summary: "turbospark-model recommend / pull <alias>. Installs go to ~/.turbospark/models/<alias>.gturbo. A failed pull restarts from zero, it cannot resume"
tags: ["models", "day-one"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e04"]
source: "docs/MODELS.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## How do I get a model to run locally?

```sh
turbospark-model recommend               # ranked for THIS machine
turbospark-model list                    # the whole curated catalog
turbospark-model info gemma4             # one row in full
turbospark-model pull gemma4             # install it
turbospark-check --model gemma4 --messages-file /tmp/p.json
```

`--model` takes either a catalog alias or a path, on both `turbospark-check`
and `turbospark-server`. An existing directory always wins over an alias of
the same name, so a bare name never silently resolves to the wrong model.

Installs land at `$TURBOSPARK_HOME/models/<alias>.gturbo` (default
`$TURBOSPARK_HOME` is `~/.turbospark`), recorded in
`$TURBOSPARK_HOME/installed.json`. `turbospark-model path <alias>` prints
the directory and fails if not installed, useful in a script that should
fail fast rather than pass an empty `--model ''`.

## Don't

- Don't retry a `pull` expecting it to resume. It can't: the ranged
  downloader retries one chunk up to 8 times then gives up, and giving up
  loses the whole walk. A pull that dies 19 GB into a 26 GB stream starts
  again from zero. Budget the full download time before starting.
- Don't read "smaller model" as "fits here." What decides whether a
  checkpoint can stream on this engine is how finely it splits its experts
  (one expert's byte size x the slot cache), not its parameter count.
  Mixtral 8x7B is smaller than several models that fit, and wants ~55 GiB of
  pinned expert-slot cache at the default 16 slots. `turbospark-model probe`
  prints the arithmetic before you download anything.
- Don't read a `recommend` row's ALLOCS and ON DISK columns as the same
  question. ALLOCS is what `open()` allocates (slot cache + KV), and
  exceeding it is a failed open. ON DISK is the whole install, and
  exceeding memory there just costs throughput since this engine streams.
- Don't quote a measured peak-memory number across a different context
  window or slot count. It's a peak at one specific (context, slots) pair.
  Gemma 4 is 2,175 MiB at 4,096/16 and 3,654 MiB at 4,096/32.

## Status tiers, read literally

`verified` means a frozen quality-gate or memory-oracle test exists and can
go red. `runs` means it installed and generated coherent text with no
frozen row. `caveat` (only `mixtral` today) means it runs, but a
disqualifying property is documented, kept in the table specifically
because that property is otherwise invisible.
