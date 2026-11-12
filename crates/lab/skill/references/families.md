# Measuring by family

Every text family and audio kind has a profile in `crates/lab/families/`
(`tslab family` prints it). The profile says what to set first, which sweep
to run first, which readings exist and what they were taken under, and the
traps. Start there before a survey:

```sh
$TSLAB family --model M                 # recommended settings and sweeps that apply to M
$TSLAB family --model M --findings      # with every recorded finding and its provenance
$TSLAB family                           # every profile, one line each
$TSLAB sweep --model M --sweep all --experiment E --json
$TSLAB workflow family-sweep --model M --experiment E --dry-run
```

A profile records, it never decides. Doc-sourced findings are `reading`s
with their heading and conditions, not tslab verdicts, and they are not to be
quoted as if tslab measured them. A `recommended` entry names the findings
behind it: read those before copying the setting.

## Text families

| Family (`as_str`) | Flow | First sweep | Known best | Trap |
|---|---|---|---|---|
| `qwen35` (Qwen 3.5/3.8 dense, GDN) | qwen | `spec_block` with an MTP install, `mma8_mode` with a DFlash2 install | MTP block 2, tile (`int4_mma8`) unset, `gdn_dk128` on (default), GQA attention (default), `prefill_small_n` on (default) | never force `attn_decode_kernel=barrier` as a baseline; the MTP install's stream already differs from plain decode on short-explanation; chunked-prefill rows ledgered before 2026-10-27 predate `prefill_small_n` (the 48-row GDN a/b projections left the scalar slices), so pair a prefill A/B against `prefill_small_n=0` in one binary and do not compare across the change |
| `qwen36` (Qwen 3.6 MoE, GDN) | qwen | `routed_batch`, then `slots` | Steel batched arm (default), `routed_batch=1` at 32+ slots, slots auto (96 here) | declare `moe_kl`; `qwen_prefetch` panics on a non-integer; the read-overlap seam refuses chunked prefill |
| `flint` | flint | `chunked_prefill` (bit-exact) | `flint_chunked_prefill` on (default since the clean +38.2% prefill win, 2026-10-28); 16 slots | `batched_gemv` and `routed_batch` refused by name; every lab run before 2026-10-28 was dirty |
| `gptOss` | gptoss | `routed_batch` | routed_batch=1 (+23.5% prefill, 2026-10-29, opt-in pending the moe_kl gates); chunk 128, 16 slots | sinks keep `rows_attn` off AND the GQA decode kernel unreachable; no INT4 bench for MXFP4 experts |
| `gemma4` | gemma4 | `routed_batch` (INT4 installs), `residency` | 16 slots with routed_batch=1 (+3.3% prefill, 2026-10-29, opt-in pending the moe_kl gates), `shared_cb` on (default; kill arm -27.4%) | two attention head shapes at head_dim 512 (the GQA decode kernel cannot serve them); the batched routed prefill refuses at 32+ slots; the residency sweep cannot pair on the current sampler; the dense Humanizer owes `dense_kl` |
| `llama`, `qwen3`, `qwen2`, `minimax_m2`, `qwen3_vl` | llama | `batched_gemv`, `decode_attention` | chunk 128 on the dense half; GQA decode default on dense llama, qwen2, qwen3 and qwen3_vl (2026-10-29 flip: +5.7% mistral7b, +3.9% qwen2, +19.8% qwen3_vl, barrier opt-out -5.4% on qwen3) | Llama and Mixtral share a string: the gate class follows `expert_count`, and Mixtral keeps barrier decode; `minimax_m2` is unmeasured and keeps barrier; BF16 trunks refuse the batched arm; head_dim-64 installs fall back to barrier in the encoder regardless |
| `qwen3moe` | llama | `slots` | 64 slots (+27.9% decode over 16, 2026-10-29; 16 is the tight-memory choice); barrier decode (default) | the GGUF install refuses `routed_batch=1`; the gqa decode arm won +3.4% but changed the decoded tokens, so it stays off until a `moe_kl` gate runs |
| `qwen4exp` | qwen4 | `slots_wired` | slots auto with wiring on (the default since 2026-10-31: 96 wired slots on a 36 GB machine, +6.7% decode over 48 unwired; 128 wired won 16.9%); the multi-row prefill seams and DFlash2 (about 1.0x priced) do not pay on real text | 2,048 context; the hyper-connections widen the residual; an MLX install decodes garbage if its RMSNorms run uncentered, which inflates hit rates and seam wins; an unwired cache above auto loses to memory compression (`expert_slots_wired=0`) |
| `museGlimmer` | museglimmer | `batched_gemv` | chunk 128 | the highest J/token in the power baseline (powermetrics); tslab's IOReport joules are a different meter and compare only within an experiment |
| `lfm3` | lfm3 | `attn_kernel` (default against barrier), `chunked_prefill` | GQA attention, INT4 chunked prefill and fused FFN all on by default; chunk 128 | the GGUF Q8_0 install has no batched GEMM, use `--chunk off`; phase buckets are final-only (one command buffer), use kernel benches or trace for shares; the fused FFN is bit-exact but NOISE on decode |
| `spark2_5`, `deepseek2` | spark, deepseek2 | `decode_attention`, `slots` | `--chunk off` (the only driver); GQA decode default on spark (2026-10-29, +3.9%); deepseek2 slots 64 (+28.1% over 16, 2026-10-29; 32 mid, 16 tight-memory) | MLA has no attention bench; Spark's fused q|k|v is one shape |
| `deepseekV4Flash` | none | none | none | no decode flow: kernel benches only |

## Audio kinds

| Kind (`model_type`) | Job | Worker arm | Known | Trap |
|---|---|---|---|---|
| `whisper` | transcribe | yes (Metal) | tiny.en 160-165 ms warm on 20 s | the CPU path refuses the worker; 16 kHz mono WAV only |
| `moonshine` | transcribe | yes | CPU 176 ms vs Metal 18 ms, 120 vs 316 MiB | CPU is the default; Metal serves the Tiny shapes only |
| `qwen3_asr` | transcribe | yes | CPU 10.2 s vs Metal 5.4 s median | the audio tower is still on the CPU |
| `minimax_music3` | music | yes | RTF readings on battery only | 60 s and 30 steps by default: use `music-short.json` |
| `kokoro` | speak | yes | none | en-US only |
| `megatts3` | speak | yes | none | `consent: true` and `reference_wav` or the arm is refused |
| `omnivoice`, `fish_drama3` | speak | no | Fish Drama 3 peak 6106-6296 MiB | kernel benches only |

Audio arms are judged on output identity (`bit_exact` for a waveform,
`text_exact` for a transcript). There is no KL class. A job's `steps`,
`duration_s` and `seed` are settings, and arms cannot change a job, so a
steps comparison is two runs with two job files.

## Recording a finding

After a sweep or an A/B on a family, add the verdict to its profile:
`crates/lab/families/README.md` gives the shape (ledger experiment and
record id, install, machine, power, revision, workload, conditions, values).
If it changes what to set first, add or edit a `recommended` entry naming it.
Numbers go in the profile or the topic doc, never in an AGENTS.md, and a
row in `docs/BENCHMARKS.md` is a human decision.
