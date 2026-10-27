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
| `qwen35` (Qwen 3.5/3.8 dense, GDN) | qwen | `spec_block` with an MTP install, `mma8_mode` with a DFlash2 install | MTP block 2, tile (`int4_mma8`) unset, `gdn_dk128` on (default), GQA attention (default) | never force `attn_decode_kernel=barrier` as a baseline; the MTP install's stream already differs from plain decode on short-explanation |
| `qwen36` (Qwen 3.6 MoE, GDN) | qwen | `routed_batch`, then `slots` | Steel batched arm (default), `routed_batch=1` at 32+ slots, slots auto (96 here) | declare `moe_kl`; `qwen_prefetch` panics on a non-integer; the read-overlap seam refuses chunked prefill |
| `flint` | flint | `chunked_prefill` (bit-exact) | `flint_chunked_prefill=1` once its paired verdict lands; 16 slots | `batched_gemv` and `routed_batch` refused by name; every lab run so far was dirty |
| `gptOss` | gptoss | `routed_batch` | chunk 128, 16 slots | sinks keep `rows_attn` off; no INT4 bench for MXFP4 experts |
| `gemma4` | gemma4 | `routed_batch` (INT4 installs), `residency` | 16 slots, `shared_cb` on (default) | two attention head shapes, read both `attn_decode` cases; the dense Humanizer owes `dense_kl` |
| `llama`, `qwen3`, `qwen2`, `minimax_m2`, `qwen3_vl` | llama | `batched_gemv`, `decode_attention` | chunk 128 on the dense half | Llama and Mixtral share a string: the gate class follows `expert_count`; BF16 trunks refuse the batched arm |
| `qwen3moe` | llama | `slots` | 16 slots | the GGUF install refuses `routed_batch=1` |
| `qwen4exp` | qwen4 | `slots` | untuned; the REAP-288 rows spread 25-60% | 2,048 context; the hyper-connections widen the residual |
| `museGlimmer` | museglimmer | `batched_gemv` | chunk 128 | the highest J/token in the power baseline (powermetrics); tslab's IOReport joules are a different meter and compare only within an experiment |
| `spark2_5`, `deepseek2` | spark, deepseek2 | `decode_attention`, `slots` | `--chunk off` (the only driver) | MLA has no attention bench; Spark's fused q|k|v is one shape |
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
