# Wiring a new model family

A checklist, in dependency order, for taking a checkpoint from "downloaded"
to "generates coherent text at a defensible memory footprint". Written from
what actually went wrong bringing up Gemma 4 26B-A4B; every "why" below is a
bug that shipped, not a hypothetical.

Read `AGENTS.md` Gotchas 16 to 19 first. They are the four traps that cost
the most time, and three of them are invisible to a greedy smoke test.

The ordering matters. Each phase has a gate that must pass before the next
phase's failures are interpretable: if you skip ahead and the output is
garbage, you will not know which of five layers to look at.

---

## Phase 0 - Decide the scope, in writing

Before touching code, answer these from the checkpoint's `config.json` and
the reference implementation. Every answer becomes a field in `ArchConfig`
or a reason to stop.

- [ ] **Layer kinds.** Which layers are full attention, sliding-window,
      linear (GDN), compressed (MLA/DSV4)? This becomes
      `full_attention_layer_mask` (1 / 0 / 2 / 3-4). If any kind's kernels
      are unported, `RealForwardRunner::open` must reject the install with
      a clear error, not produce wrong numbers.
- [ ] **Per-layer shape divergence.** Gemma 4 uses `head_dim` 256 on SWA
      layers and `full_head_dim` 512 on full ones, with different KV head
      counts and different RoPE thetas. Assume divergence until you have
      checked; a single `head_dim` is the exception, not the rule.
- [ ] **Attention scale.** Do NOT assume `1/sqrt(head_dim)`. Gemma 4's is
      `1.0` (the query norm absorbs it). It is a manifest field with a
      family baseline fallback; get it from the reference implementation,
      not from the formula you remember.
- [ ] **Normalization inventory.** Learned vs no-scale, per-head vs
      per-tensor, pre- vs post- vs sandwich. Write the layer flow out as
      pseudocode before implementing it; `real_forward_gemma4.rs`'s module
      header is the format to copy.
- [ ] **Output head.** Tied embeddings? Logit softcap? What does the
      reference's `forward()` RETURN - raw logits, capped logits, or
      probabilities? (See Phase 4; this is the single highest-risk line.)
- [ ] **MoE shape.** Expert count, top-k, router quantization, per-expert
      scales, shared expert, routed-weight normalization (softmax over
      top-k vs softmax over all then renormalize - these differ and both
      exist in the wild).
- [ ] **Chat template and EOS set.** From `chat_template.jinja` and
      `generation_config.json`. `eos_token_id` is often a LIST.

Gate: you can describe one decoder layer as ten lines of pseudocode without
looking anything up.

---

## Phase 1 - Repack, and prove the repack alone

- [ ] Map every checkpoint tensor name to a resident-index entry. Keep the
      source's verbatim naming if the runner selects its decode flow by
      naming (`real_forward.rs` vs `real_forward_gemma4.rs` do exactly
      this).
- [ ] Write the manifest's `arch` object with every shape field explicit.
      Optional family-extension fields fall back to the family baseline in
      `model_io::arch_baselines`, so anything that differs from the
      baseline MUST be written out.
- [ ] Add a synthetic install builder next to
      `build_synthetic_gemma4_real_install`: deterministic untrained
      weights, the real naming, the real quantization tags, small enough to
      run in CI. This is what every later test drives.
- [ ] Extend `repack::manifest_peek::peek_arch` if the family needs new
      fields. Resolve arch in ONE place so the CLI, the bench harness, and
      the tests cannot drift apart on a fallback default.

Gate: `cargo test -p mrefrust-repack` passes, and the synthetic install
opens through `RealForwardRunner::open` without touching decode.

---

## Phase 2 - Kernels, each parity-tested in isolation

- [ ] For each new kernel, vendor the MSL and write a CPU reference in
      `crates/compute` if one does not exist. A kernel with no reference is
      a kernel you cannot verify; `logit.metal`'s `sample` was descoped for
      exactly this reason.
- [ ] Add a parity test in `crates/gpu/tests/` against that reference on
      real hardware. Cover the saturating / wrapping / edge inputs, not
      just the middle of the range.
- [ ] Port-local kernels (no Swift original) are allowed - `scalar_mul_fp16`
      and `logit_softcap_fp16` are two - but the shader comment must say
      so and say why the fused upstream form does not fit.
- [ ] **Function constants are part of the pipeline cache key.** If you
      specialize a value into a pipeline, its bytes must go into
      `constants_key`, or the first dispatch's specialization gets reused
      for every later one. `encode_attention_decode` keys on both `scale`
      and `ring_capacity`.
- [ ] Watch for constants the shader checks UNCONDITIONALLY (no
      `is_function_constant_defined` gate). Specializing those with a dummy
      value silently overrides the runtime buffer argument.

Gate: `cargo test -p mrefrust-gpu` passes on the Metal device.

---

## Phase 3 - The decode flow, greedy first

- [ ] Implement the layer loop against the Phase 0 pseudocode. Keep the
      pseudocode in the module header and keep it accurate.
- [ ] Size the KV cache from the layer mask. **Enable the SWA ring**
      (`KvCacheManager::new`'s `fp16_ring_enabled`) whenever the mask has
      sliding-window layers, size them
      `min(max_context, sliding_window + prefill_chunk)`, and pass
      `ring_capacity(layer)` into `encode_attention_decode`. A comment
      claiming "all-full-attention, no ring needed" is how 600 MiB of KV
      got allocated for nothing; derive the flag from the mask, not from
      prose. The capacity must also reach the pipeline-cache constants
      key, or ring dispatches reuse the linear pipeline.
- [ ] Wrap the per-token entry point in `gpu::autorelease_pool`. Not
      optional - see Gotcha 17.
- [ ] Preallocate all activation scratch at open. Assert the hot path
      allocates no Metal buffers (`gpu_buffer_allocations()` flat across
      tokens); copy the existing test.

Gate: greedy generation on the real checkpoint, with the chat template
applied, produces coherent text for 400 tokens. If it does not, the bug is
in the math, and nothing after this phase is worth debugging.

---

## Phase 4 - The output head and the sampler boundary

This phase gets its own section because it is where a model that looks
perfect under greedy decoding is quietly broken.

- [ ] **`produce` writes logits. The sampler softmaxes. Once.**
      `selection::select` normalizes whatever it receives. A head that also
      normalizes yields `softmax(softmax(z))`, which over a large vocab is
      nearly uniform - and because softmax is monotone, ranking survives,
      so greedy output is *byte-identical to correct*. Only sampling
      exposes it, as fluent text that derails into word salad after a few
      dozen tokens, with stray unused/foreign tokens sprinkled early.
- [ ] Apply the logit softcap in the head if the family has one (that is
      what HF's `*ForCausalLM.forward` returns), and stop there. If you
      need a fused cap+softmax kernel for a GPU sampler later, add it then.
- [ ] Check the softcap survives FP16 before trusting it. Dump the top-8
      raw logits for a few real tokens: if they sit deep in tanh
      saturation, many will round to the same FP16 value and ties will
      collapse to the lowest index. Gemma 4's peak around 34-38 caps to
      ~24.5, comfortably clear.
- [ ] Resolve the full EOS set from `generation_config.json` (it is a
      list), and confirm generation actually STOPS. `stop reason
      MaxTokens` on a short question is a symptom, not a setting.

Gate: SAMPLED generation at the CLI defaults stays coherent for 400 tokens,
and a short question terminates with `EndOfTurn`. Run this even when greedy
already looks perfect - especially then.

---

## Phase 5 - Memory

- [ ] Write the static accounting down first: resident weights + KV +
      expert slot capacity + process baseline. Compute each from shapes,
      not from a measurement.
- [ ] The resident weight mapping COUNTS in `phys_footprint`. A plain
      read-only `mmap` would not, but `newBufferWithBytesNoCopy` pins it.
- [ ] Run `crates/bench/tests/memory_oracle.rs` against the install. It
      asserts two things:
      - peak <= accounting, with no slack (catches an allocation the
        design does not describe);
      - replaying one already-warm case stops growing (catches anything
        that accumulates per token, which the ceiling would hide under
        unused expert slot capacity until it exceeded ~500 MiB).
- [ ] If the steady-state guard fails, `vmmap --summary <pid>` on a live
      run attributes it by region in seconds: `IOAccelerator (graphics)`
      is Metal buffers, `MALLOC_LARGE` with a region count near
      `slots x layers` is the expert slot cache, and a linear per-token
      climb with a flat `gpu_buffer_allocations()` is an autorelease pool
      you forgot.

Gate: the oracle passes, and the peak is in the same neighbourhood as the
reference implementation's published number for the same workload.

---

## Phase 6 - Write it down

- [ ] `AGENTS.md`: a Gotcha for anything a reader would get wrong twice.
- [ ] `DEVIATIONS.md`: every deliberate divergence from the reference, with
      the reason. "Swift fuses cap+softmax because it samples on the GPU"
      is the shape - what they do, what we do, why the difference is
      correct.
- [ ] Module headers: the layer-flow pseudocode, the memory model, and the
      output contract. Anyone debugging at 3am reads these first.
- [ ] Update this file if a new trap cost you more than an hour.

---

## Quick triage table

| Symptom | Look here first |
|---|---|
| Greedy fine, sampled degenerates | Double softmax in the head (Gotcha 16) |
| Babble from token 1 on any prompt | Chat template not applied |
| Coherent then collapses at a fixed position | KV ring capacity vs window |
| Never emits EOS | EOS set not resolved from `generation_config.json` |
| Memory grows linearly with tokens | Missing autorelease pool (Gotcha 17) |
| Memory 3-4x the reference at open | SWA layers sized at `max_context` |
| Output changed after a kernel tweak | Function constant missing from the pipeline cache key |
| "No output" from `mference-check` | `--messages-file` is not wired; use `--prompt` |
