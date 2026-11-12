# turbospark-lab Blueprint

## System Overview
`crates/lab` (package `turbospark-lab`, not published) builds `tslab` and `tslab-worker`: per-family phase profiles, gated interleaved A/B with a statistical verdict, shape-derived kernel benches, and an append-only ledger [Cargo.toml:bin]. The optional `ane-prefill` feature enables the ANE+Steel arm [Cargo.toml:features]. Beside the Rust source it ships a skill package, a trace helper script, and data (pinned settings, workloads, family profiles, `program.md` for the autoresearch loop) that carry no contract.

## Subdirectories & System Boundaries
- src: [./src/EGM.md](./src/EGM.md) - the library, commands, worker, ledger, gates.
- src/bin: [./src/bin/EGM.md](./src/bin/EGM.md) - `tslab` and `tslab-worker` entry points.
- src/dash: [./src/dash/EGM.md](./src/dash/EGM.md) - read-only dashboard (`tslab serve`).
- src/kernel: [./src/kernel/EGM.md](./src/kernel/EGM.md) - kernel microbenches.
- skill: [./skill/EGM.md](./skill/EGM.md) - agent skill package.
- skill/scripts: [./skill/scripts/EGM.md](./skill/scripts/EGM.md) - doctor and preflight shell checks.
- tools: [./tools/EGM.md](./tools/EGM.md) - `trace_tables.py` Metal trace summarizer.

## Global Invariants
- Orchestrator decides, worker measures, one fresh cleared-env process per run [src/orchestrate.rs:PASSTHROUGH].
- Evaluator inputs are protected from the experiments they judge [src/integrity.rs:FORBIDDEN_PREFIXES].
- Ledger and result files are append-only or write-once [src/ledger.rs:Ledger] [src/atomic.rs:write_new].
- Real measuring code is macOS-only; the portable half builds everywhere [src/lib.rs:cli].

## Global Side Effects
- Writes the lab dir (`~/.turbospark/lab` by default) and takes `.machine.lock` [src/paths.rs:lab_dir] [src/lock.rs:MachineLock].
- Uses the Metal GPU, Core ML, `git`, and `xctrace` [src/trace.rs:trace_cmd].

## System Verification
- Run: `cargo test -p turbospark-lab`
