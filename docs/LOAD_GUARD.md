# Load guardrails, the AutoFit floor, and the pressure watcher

Three related knobs that decide how much of a machine this engine will
commit to loading a model, how small an automatically-sized context window
is allowed to get, and what the decode loop does when the machine comes
under strain while a turn is in flight.

Read `docs/DECODE_BUDGET.md` for what a slot count buys and
`docs/BENCHMARKING.md` for how the frozen footprint rows were taken. This
page is the home for the POLICY; neither of those changes.

## The one thing to know first

**`relaxed` is the default and it is exactly what shipped before any of this
existed.** Every frozen peak in `docs/BENCHMARKS.md`, every `measured` block
in `crates/catalog/src/models.json`, and the memory oracles' ceilings all
describe an engine budgeting at a 4 GiB reserve and a quarter of what is
left. A default that resolved to anything else would not fail anywhere: it
would leave every one of those rows quietly describing a configuration the
engine no longer opens with.

That is asserted rather than claimed, in three places:

- `load_guard_tests::relaxed_is_exactly_todays_arithmetic` pins the three
  constants.
- `context_policy_tests` passes `LoadPolicy::default()` in all twelve
  pre-existing cases with their original expected numbers, so the file
  staying green IS the no-change proof for the context resolver.
- `fit_tests::the_default_guard_reproduces_the_frozen_thresholds` pins the
  tight threshold across the crate boundary, which `model_io` cannot state
  itself because it cannot import `catalog`.

Change a tier's numbers and those are the tests that decide whether you have
moved a user-facing knob or invalidated a published measurement.

## The tiers

| Tier | Reserve | Share of the rest | Tight at | Refuses |
|---|---|---|---|---|
| `off` | 0 | all of it | 90% | no |
| `relaxed` (default) | 4 GiB | 1/4 | 90% | yes |
| `balanced` | 8 GiB | 1/6 | 80% | yes |
| `strict` | 12 GiB | 1/10 | 70% | yes |
| `<bytes>` (custom) | 4 GiB | 1/4 | 90% | yes, plus a hard cap |

The tiers are ORDERED: anything `strict` admits, `balanced` admits, and so
on up to `off`. Nothing about three independent fields enforces that, so it
is swept in `the_tiers_are_ordered_from_off_down_to_strict` rather than left
to the reader -- and `the_tiers_are_distinguishable_on_a_real_machine` sits
beside it, because a sweep over tiers that had all collapsed onto one set of
numbers would pass the ordering test on every input. That pair was mutation
checked: collapsing `balanced` onto `relaxed` reddens the second and not the
first.

**`off` is not "a very large budget".** A budget alone still refuses a
request past it, and this tier exists to not do that. It reserves nothing,
spends the whole pool, and declines to refuse -- so a caller who names a
window too large for the machine gets the Metal allocation failure they
asked for rather than a message. It still TIERS, because a caller wants to
know a fit is tight even when nothing will stop them.

**Custom's ceiling is on what the engine ALLOCATES, not on the install's
size**, and that is the whole reason it is expressed in `counted` bytes.
`crates/catalog/src/recommend/fit.rs`'s module header has the argument:
exceeding memory with the MAPPED install is the streaming this engine is
built around and costs throughput rather than correctness. A cap read
against the install size would refuse a 13 GB model on a 16 GB machine,
which runs, and which the slot policy's floor exists for.

## The AutoFit floor

`--min-auto-context N` refuses to open when an AUTOMATIC window resolves
below `N` tokens. Zero, the default, imposes no floor.

**It is scoped to `auto` and says nothing about an explicit
`--max-context`.** A caller naming a number has decided how to spend their
own machine; refusing it for being under an AutoFit minimum would be the
setting acting outside what its name claims. That is the same distinction
`context_policy` already draws twice -- between the trained-context WARNING
and the memory REFUSAL.

**It applies under every tier including `off`**, because it is not a memory
precaution the guard could relax. It is the caller stating a requirement of
their own workload.

The refusal names which of three bounds was binding, because lowering a
floor, loosening a guard and picking a different checkpoint are three
different actions and only one of them helps:

- `ContextCap::Memory` -- a looser guard or freeing memory raises it.
- `ContextCap::Trained` -- the checkpoint was never trained that far, so no
  tier helps.
- `ContextCap::Undeclared` -- the install declares no trained context, so
  `auto` fell back to the default window; name an explicit `--max-context`.

## The watcher

The decode loop already polled thermal pressure every 16 tokens and stepped
its rate cap down under it. Memory pressure now rides that same block.

**It reads the kernel's own three-valued verdict
(`kern.memorystatus_vm_pressure_level`: 1 normal, 2 warn, 4 critical) and
never a free-page count.** `crates/gpu/src/power_state.rs` declines
`host_statistics64`'s free pages for the sizing policies on the grounds that
a number moving second to second makes two runs incomparable. That is
correct and it is about a BUDGET, which has to be stable across opens or no
two footprints can be compared. A watcher is the opposite job: it exists to
see the thing that moves. Reading a verdict rather than a page count keeps
the two from being the same instrument, and nothing budgets from it.

`0` from that sysctl means the probe did not answer, and it maps to
`Normal`. That is the OPPOSITE of `ThermalLevel::from_raw`'s clamp, which
reads an unknown value ABOVE the range as hotter still -- deliberately: an
unrecognized thermal level means more heat, while an unrecognized memory
level means the kernel said nothing, and pacing a decode loop on the
strength of a failed syscall would be pacing on no information.

**The two ladders combine as a MINIMUM, not a precedence.** The signals are
independent: a machine can be cool and short of memory (something else just
opened a model) or hot and comfortable. Asking which one "wins" is the wrong
question, since each states a ceiling true on its own terms, and honouring
the looser of two true ceilings would ignore one of them.

The watcher follows the power profile's stepping rather than having its own
switch, so `performance` (the default, and what every published benchmark
was measured under) polls nothing and the decode loop executes exactly the
statement sequence it did before. `balanced` and `efficiency` watch both.

### What it does NOT do

**It does not unload anything, and that is a contract rather than an
omission.** The engine caps its own decode rate; it does not close sessions,
because it does not own them. The FFI's handle belongs to the caller
(`crates/ffi/CLAUDE.md` Gotcha 1), and a session that destroyed itself would
leave every host holding a dead pointer it never asked to be given.

What the engine does instead is REPORT, in two places:

- `RawDecodeResult.peak_memory_pressure` -- the worst level seen during the
  turn. A watcher that only exposed the level at the moment a caller asked
  would miss a spike, and a spike is the whole event worth telling a host
  about.
- `ts_system_info_json` -- polled on demand, unconditionally, so a host that
  runs `performance` (and therefore has no in-loop probe) can still see the
  machine's state.

`Normal` on a result therefore means "no reading was taken" as often as it
means "memory was fine". A host that needs the second should ask the
telemetry call.

## Files

| File | What lives there |
|---|---|
| `crates/model-io/src/load_guard.rs` | `LoadGuard`, `GuardBudget`, `LoadPolicy`. Pure, portable, no OS probe. |
| `crates/model-io/src/load_guard_tests.rs` | Tier ordering, distinguishability, the default pin. |
| `crates/model-io/src/context_policy.rs` | `resolve_max_context` reads the budget; `ContextRefused`, `ContextFloorUnmet`, `ContextCap`. |
| `crates/catalog/src/recommend/fit.rs` | `fit()` reads the same budget, so a recommendation and the open it recommends cannot disagree. |
| `crates/gpu/src/power_state.rs` | `memory_pressure_raw()`, the sysctl. Here because `runtime` is `#![forbid(unsafe_code)]`. |
| `crates/runtime/src/power.rs` | `MemoryPressure`, `memory_cap`, `thermal_cap`, `stepped_cap`, `RateControl.memory_probe`. |
| `crates/runtime/src/raw_completion.rs` | The one poll block both signals share. |
| `crates/invocation/src/request.rs` | `LoadGuard`, the pure parser's mirror. `crates/cli` maps between the two. |
| `crates/ffi/src/wire.rs` | `load_guard()`, `RecommendOptions`, the two `OpenOptions` fields. |

**`crates/invocation` carries its own `LoadGuard` on purpose.** That crate is
pure and may not read a machine or an install, and every tier is a claim
about one. `map_load_guard` in `crates/cli/src/generate/session.rs` is the
single place the two meet, the same shape `map_power_profile` already has.

## Surfaces

```sh
# CLI. A tier word or a byte ceiling, one flag: a separate --load-guard-bytes
# would let a caller name a tier and a ceiling that disagree.
turbospark-check --model gemma4 --messages-file /tmp/p.json \
  --load-guard balanced --min-auto-context 8192

# Server, same grammar, resolved once at startup.
turbospark-server --model gemma4 --load-guard strict

# The hub's own arithmetic, under the tier the sessions will open with.
turbospark-model recommend --context 8192 --load-guard balanced
```

```swift
var options = OpenOptions()
options.loadGuard = .balanced          // or .off, .relaxed, .strict, .custom(bytes)
options.minAutoContext = 8192          // 0, the default, imposes no floor
```

```c
/* NULL, "" and "{}" all mean every default, which is "relaxed". */
int32_t ts_recommend_json(uint32_t context_window, const char *options_json,
                          char **out);
```

## The trap when adding a caller

**A recommendation and the `open()` it recommends must share a tier.** They
share a budget by construction today, which is what makes a recommendation
trustworthy. A hub ranking under `relaxed` while its sessions open under
`strict` promises a fit the loader then refuses, in the one place a user has
no way to see the two disagree.

That is why `catalog::Machine` carries `load_guard` rather than `fit()`
taking it as a loose parameter, why `Fit` carries the tier that produced it
(so a caller substituting a measured peak and re-deriving the verdict cannot
silently apply a different one), and why `ts_recommend_json` grew an options
argument rather than being left on the default.

## See also

- `docs/DECODE_BUDGET.md` -- what a slot count buys, and the three decode
  dead ends.
- `docs/BENCHMARKING.md` -- how peak footprint is measured, and the memory
  oracle.
- `docs/POWER_BASELINE.md` -- the thermal half of the ladder, measured.
- `docs/SWIFT_BINDINGS.md` -- the ABI contract and the Swift surface.
