# Wiring a new model family

A checklist, in dependency order, for taking a checkpoint from "downloaded"
to "generates coherent text at a defensible memory footprint". Written from
what actually went wrong bringing up Gemma 4 26B-A4B; every "why" below is a
bug that shipped, not a hypothetical.

Read `AGENTS.md` Gotchas 16 to 19 first. They are the four traps that cost
the most time, and three of them are invisible to a greedy smoke test.
Gotchas 24 to 26 are the family-specific ones: the manifest's
family-extension fields, fixed threadgroup contracts, and tensor names that
break the `.weight` convention. 24 blocks Phase 1 outright.

The ordering matters. Each phase has a gate that must pass before the next
phase's failures are interpretable: if you skip ahead and the output is
garbage, you will not know which of five layers to look at.

Every command a gate below names in prose is spelled out in `AGENTS.md`:
"Build, test, dev commands" for the per-crate tests and the memory oracle,
"Real-model smoke" for the two 400-token runs Phase 3 and Phase 4 gate on.

**A new source is a different axis from a new family, and this file is about
the family axis.** Bringing GGUF in (ROADMAP Phase G) changed Phase 1,
Phase 2 and Phase 7's probe, and touched nothing in Phases 3 to 6: the
decode flow, the head, the memory model and the quality gates do not know
where the bytes came from.
So if you are adding a source rather than an architecture, read Phase 1,
Phase 2 and Phase 7 and skip the rest. The differences are marked "source:" below, and
the record is `crates/repack/CLAUDE.md` Gotchas 4 to 7 plus `AGENTS.md`
Gotchas 29, 30 and 33. The one thing that axis adds and this one does not
have is that a source can be right about every name and still wrong about
what a tensor means (Gotcha 33).

---

## Phase 0 - Decide the scope, in writing

Before touching code, answer these from the checkpoint's `config.json` and
the reference implementation. Every answer becomes a field in `ArchConfig`
(`crates/model-io/src/arch_config.rs`; family baselines in
`crates/model-io/src/arch_baselines.rs`) or a reason to stop.

- [ ] **Source: where does `ArchConfig` come from, and how is the family
      identified?** A safetensors checkpoint answers both from
      `config.json`. A GGUF has no `config.json`: the values come from
      metadata keys under an architecture prefix (`gguf_config.rs`), and
      several behavioural fields are simply absent because llama.cpp
      hardcodes them in its graph builder, so the derivation starts from
      `known_architecture(family)` and overrides only what the file really
      determines. Worse, `general.architecture` is the converter's name and
      not the family's -- Qwen 3.6 GGUFs say `qwen35moe` -- so deriving it
      from `ModelFamily::as_str()` recognizes no real file. Whatever the
      source, the strongest available check is the same: assert the derived
      `ArchConfig` equals the one an independently produced install of the
      same model declares (`gguf_checkpoint_network.rs`). The two sides
      share no code and no input.

- [ ] **Layer kinds.** Which layers are full attention, sliding-window,
      linear (GDN), compressed (MLA/DSV4)? This becomes
      `full_attention_layer_mask` (1 / 0 / 2 / 3-4). If any kind's kernels
      are unported, `RealForwardRunner::open`
      (`crates/runtime/src/real_forward.rs`) must reject the install with
      a clear error, not produce wrong numbers.
- [ ] **Per-layer shape divergence.** Gemma 4 uses `head_dim` 256 on SWA
      layers and `full_head_dim` 512 on full ones, with different KV head
      counts and different RoPE thetas. Assume divergence until you have
      checked; a single `head_dim` is the exception, not the rule.
- [ ] **Three one-line flags that each change the layer graph**, all
      manifest fields with a Gemma fallback, so all three must be answered
      even when the answer is "same as Gemma". `attention_k_eq_v`: Gemma's
      full layers take V from the K projection (`crates/runtime/src/families/gemma4/mod.rs`
      still writes and per-head-norms a separate V row from those weights;
      the short-name flow in `real_forward.rs` goes further and binds the
      K buffer directly as V, so its V buffers are never written and its
      pages never become resident, and it rejects `attention_k_eq_v ==
      false` outright). Qwen has a real, separate `v_proj`.
      `embedding_scaled_by_sqrt_hidden`: Gemma yes, Qwen no.
      `ffn_sandwich_norms`: Gemma normalizes the attention and FFN outputs
      before adding them back, Qwen adds them raw.
- [ ] **Attention scale.** Do NOT assume `1/sqrt(head_dim)`. Gemma 4's is
      `1.0` (the query norm absorbs it); Qwen 3.6's IS `256^-0.5`. Both
      exist, so the formula is never evidence. It is a manifest field with
      a family baseline fallback; get it from the reference implementation,
      not from the formula you remember.
- [ ] **RoPE convention, both halves of it.** "Partial rotary" names at
      least two different transforms and this port ships both. Gemma's
      `rope_proportional_neox` rotates a prefix of the pairs across the
      full head, pairing `(i, head_dim/2 + i)`, dividing frequencies by
      `head_dim`. Qwen's `rope_neox_subdim` rotates all the pairs of a
      prefix of the head, pairing `(i, rotary_dim/2 + i)`, dividing by
      `rotary_dim`. Same `partial_rotary_factor`, different element sets
      and different angles. Pin the pair partner and the divisor from the
      reference implementation, separately, and parity-test that elements
      past `rotary_dim` come back untouched.
- [ ] **Packed projections.** Does any projection emit more rows than its
      name implies? Qwen's `q_proj` emits `2 * num_heads * head_dim`:
      per-head `[query; gate]` pairs that must be split
      (`split_q_gate_fp16`) before the per-head norm, RoPE, or attention
      sees them. `attn_output_gate` is the manifest flag. Sizing the GEMV
      at `num_heads * head_dim` reads half the rows and silently drops the
      gate.
- [ ] **Normalization inventory.** Learned vs no-scale, per-head vs
      per-tensor, pre- vs post- vs sandwich. Count how many distinct norms
      feed the FFN branches: Gemma 4 splits three ways (no-scale for the
      router, `pre_feedforward_layernorm` for the shared expert,
      `pre_feedforward_layernorm_2` for the routed ones), Qwen 3.6 feeds
      all three from one `post_attention_layernorm`. Also count what the
      attention side normalizes: Gemma norms q, k and v per head; Qwen
      norms q and k only. Adding a v norm "by analogy" is silent. Write
      the layer flow out as pseudocode before implementing it;
      `crates/runtime/src/families/qwen/mod.rs`'s module header is the
      format to copy.
- [ ] **Output head.** Tied embeddings? Logit softcap? What does the
      reference's `forward()` returns raw logits, capped logits, or
      probabilities? (See Phase 4; this is the single highest-risk line.)
- [ ] **MoE shape.** Expert count, top-k, router quantization, per-expert
      scales, shared expert, routed-weight normalization (softmax over
      top-k vs softmax over all then renormalize - these differ and both
      exist in the wild). Two absences count as answers: Qwen has neither
      `router.scale` nor `router.per_expert_scale`, so the INT8 router
      kernel (which takes an effective-scale vector regardless) gets a
      buffer of ones and the top-k reduces to softmax-over-the-selected.
      And is the shared expert gated? Qwen's output is scaled by
      `sigmoid(shared_expert_gate(x))`, one scalar logit from its own
      1-row GEMV.
- [ ] **Expert granularity, which is one multiplication and decides whether
      the checkpoint fits this engine at all.** The slot cache is
      `slots x layers x expert_stride`, so what matters is the size of ONE
      expert, not the size of the model: `3 x moe_intermediate x hidden`
      at the expert block type. Gemma 4 is 128 experts of ~3.2 MiB and 16
      slots over 30 layers pin 1.5 GiB; Mixtral 8x7B is 8 experts of
      108.9 MiB and the same 16 slots want 54.5 GiB. Both numbers come off
      the header (`expert_count`, `feed_forward_length`) before any
      download. Do this here, next to the layer graph -- ROADMAP Phase M2
      did not, and spent a 26 GB download finding out. AGENTS.md Gotcha 36.
      **Compute the whole product, not the expert size.** `num_layers` is
      the factor that is easy to skip because it is not about experts at
      all, and it is the one that moved next: Qwen3-30B-A3B has a smaller
      expert than Gemma 4 (~2.9 MiB against ~3.2) and a larger working set
      (2,094 MiB against ~1,500), because it is 48 layers deep against 30.
      It streams and it is a good fit; it just does not land in the
      1.6-2.2 GiB band, and a bring-up should say which of those two
      outcomes it is expecting before the download rather than after.
- [ ] **`top_k_experts` against the DECODE KERNEL's own fixed reduce width --
      a THIRD capacity question, separate from expert granularity above.**
      Granularity asks whether the slot CACHE holds enough experts at once;
      this asks whether the MoE decode kernel's phase-2 down-reduce can even
      iterate that many slots in a single dispatch, which is a property of
      the KERNEL rather than of memory, and the two can disagree (a
      checkpoint can fit the cache easily and still exceed the kernel).
      `gpu::MAX_STREAMED_EXPERTS` (`crates/gpu/src/moe_decode.rs`) is the
      vendored INT4-affine pair's ceiling; every family through `gpt-oss`
      topped out at `top_k_experts = 8`, so nothing had ever forced it to be
      distinguished from "the widest top_k anyone has shipped" until
      `qwen4_exp`'s REAP-288 declared 10. Read the checkpoint's own
      `top_k_experts` against this ceiling before the download, exactly as
      the granularity bullet reads `expert_count`/`feed_forward_length` off
      the header -- a checkpoint that exceeds it is refused at `open()` with
      "top_k {} exceeds the {}-slot MoE kernels" rather than producing a
      wrong answer, which is Phase 0's field-by-field vetting doing its job,
      not a bug to work around quietly.
      **If you have to widen the ceiling, do not dispatch every family at
      the new width.** `crates/gpu/CLAUDE.md` Gotcha 13 has the full
      account: the naive fix (mask a fixed dispatch, the way
      `moe_phase2_down_reduce_k8_mxfp4` already does for `gpt-oss`'s
      top-4-of-32) would have DOUBLED phase-2's GPU cost on every top_k=8
      family that predates the new one, because their top_k had always
      equaled the old ceiling exactly, with zero padding either way. The fix
      that shipped instead sizes BOTH the dispatch width and the reduce
      loop to the checkpoint's OWN `top_k` at runtime (mirroring how phase 1
      already worked), so a top_k=8 caller is untouched and only the new,
      wider checkpoint pays for its own width. **This ceiling is per KERNEL,
      not global**: the GGUF-sourced sibling pairs (`moe_gguf.metal`'s
      Q8_0/Q4_K/Q6_K/IQ4_NL/MXFP4 kernels) each carry their OWN separately
      hardcoded reduce width and are untouched by raising
      `MAX_STREAMED_EXPERTS` -- a family arriving via GGUF with a wide
      `top_k` needs its own kernel widening, not a constant bump, and a
      test that reuses the vendored kernel's ceiling as a proxy for a GGUF
      kernel's own fixed width will silently start testing past what that
      kernel can do (exactly what happened to `moe_gguf_parity.rs`'s
      `all_eight_*_slots_participate` cases the day `MAX_STREAMED_EXPERTS`
      first moved off 8 -- five tests failed with wildly wrong sums, fixed
      by giving the GGUF siblings their own named constant rather than
      sharing the vendored kernel's).
- [ ] **Recurrent per-layer state.** Anything that is not KV: a linear
      layer's delta-rule `S`, a causal-conv tail, an SSM hidden state. For
      each, its shape, whether it grows with context (GDN's does not, which
      is the point), and its zero/empty-context value. This becomes a
      `LinearAttentionConfig`-shaped block in `ArchConfig`, a state manager
      in `crates/gpu`, and a `reset()` obligation in Phase 3.
- [ ] **How long does it answer for, not just how long is the prompt?** If
      the model reasons before answering, the shared 1,024-token budget
      truncates it and every gate that asserts `endOfTurn` fails. Measure the
      completion on the real checkpoint before freezing a row: `muse_glimmer`
      was put in the shared group on a correct prompt tokenization (46 / 408 /
      2,764) and a wrong assumption about output, and needs 1,054 / 1,378 /
      1,246 greedy and more when sampled. Note also that the rendered prompt
      is longer than the raw prose tokenizes to -- 102 against 46 here, the
      difference being the template's system preamble.
- [ ] **Chat template and EOS set.** From `chat_template.jinja` and
      `generation_config.json`. `eos_token_id` is often a list. Dialect
      resolution and the stop set live in `crates/tokenizer`
      (`MfTokenizer`, `StopMatcher`), and the dialect is resolved from the
      checkpoint's special tokens, not from the family name. The tokenizer
      files ship inside the install directory: `open_session`
      (`crates/cli/src/generate.rs`) calls
      `MfTokenizer::load_from_dir(model_dir)` on the same path it peeks the
      manifest from, so the repack has to copy them across (or the caller
      does) or every CLI/server run fails at load with no decode attempted.

Gate: you can describe one decoder layer as ten lines of pseudocode without
looking anything up.

---

## Phase 1 - Repack, and prove the repack alone

- [ ] Map every checkpoint tensor name to a resident-index entry (the
      mapping to copy: `crates/repack/src/gemma4_checkpoint/`, which is
      family-parameterized -- `classify_for_family` and `manifest_quant`
      take a `ModelFamily`, so a new family is usually a routed-expert
      marker plus four quant probe names, not a new file). Keep the
      source's verbatim naming. Two traps that cost real time on Qwen 3.6:
      a routed-expert marker the classifier does not recognize makes every
      expert a resident tensor (loads fine, generates fine, footprint
      explodes -- test it explicitly), and some parameters carry no
      `.weight` suffix at all (`linear_attn.A_log`, `linear_attn.dt_bias`).
- [ ] If this checkpoint comes from a different producer than the one the
      flow was written against (a GGUF where the port was built on
      mlx-community, say), budget a pass for source conventions before
      trusting any output. A name mapping being right does not mean a
      tensor means the same thing: llama.cpp interleaves Qwen's V heads and
      stores `-exp(A_log)` where the MLX checkpoint stores `A_log`. Undo it
      at repack time, keyed by canonical name, never at runtime and never
      in a kernel (`gguf_checkpoint/transcode.rs::v_head_axis`). Enumerate by the
      dimension the convention indexes, not by the tensors you can most
      easily compare -- that mistake made this eight tensors instead of
      three and cost a whole session (AGENTS.md Gotcha 33). Verify by
      patching a built install in place rather than repacking per attempt;
      `open()` runs no checksum, so it is seconds against ~21 minutes.
- [ ] Write the manifest's `arch` object with every shape field explicit
      and every family-extension field explicit.
      `crates/model-io/src/arch_validation.rs` resolves omitted optional
      fields against the Gemma baseline whatever family the manifest
      claims, so an install that omits them can never validate. That is
      why `gturbo_writer.rs::build_manifest_json` writes all of them
      unconditionally; do not make any conditional.
      Float fields are compared with `!=` on `f64`, and serde_json's
      default parser is accurate only to ~1 ULP, so a value that is not a
      binary fraction cannot survive the round trip. Pick powers of two
      for anything you invent for a synthetic fixture.
- [ ] Add a synthetic install builder next to
      `build_synthetic_gemma4_real_install` / `build_synthetic_qwen_gdn_moe_install`
      (`crates/repack/src/synthetic_real.rs`, `synthetic_qwen.rs` -- the
      latter reuses the former's `pub(crate)` tensor helpers, so a third
      family should too): deterministic untrained
      weights, the real naming, the real quantization tags, small enough to
      run in CI. This is what every later test drives.
- [ ] Extend `peek_manifest_arch` (`crates/repack/src/manifest_peek.rs`)
      if the family needs new
      fields. Resolve arch in ONE place so the CLI, the bench harness, and
      the tests cannot drift apart on a fallback default.

There are TWO places an `ArchConfig` comes from and both need writing: the
checkpoint's own `config.json` at repack time and the written install's
`manifest.json` at load time (`peek_manifest_arch`). A synthetic fixture
only needs the second. Skipping the first is a legitimate way to split the
work across sessions, as Qwen 3.6 did, but say so explicitly: until it
exists nobody can repack the real checkpoint.

For the first, there are two models to copy and they differ in almost every
key name, which is the point: `parse_gemma4_config`
(`crates/repack/src/gemma4_checkpoint/config.rs`) and `parse_qwen_gdn_moe_config`
(`crates/repack/src/qwen36_config.rs`). Read both before assuming a key
generalizes. Only one thing was common to them: the `text_config` wrapper,
and that is a multimodal-checkpoint convention, not a universal one.
Everything else moved -- `hidden_act` vs `hidden_activation`,
`num_experts_per_tok` vs `top_k_experts`, a flat `rope_parameters` vs one
sub-object per attention kind, `shared_expert_intermediate_size` standing in
for an `intermediate_size` the text config does not carry at all, and a
`layer_types` vocabulary whose mask codes are 2/1 rather than 1/0.

Write the parser's headline test as ONE equality against the family's
`arch_baselines.rs` entry, with the real production values inline
(`crates/repack/tests/qwen36_config.rs`). A tiny synthetic fixture cannot
catch a key wired to the wrong field, because every dimension in it is a
made-up number either way; the baseline comparison catches it immediately,
and it runs without the network.

Two fields have no config key at all and have to come from the reference
implementation rather than a formula: `attention_scale` (mlx-lm's
`Qwen3NextAttention.__init__` sets `head_dim ** -0.5`; see Phase 0) and
the family-constant booleans (`router_scaled`, `ffn_sandwich_norms`,
`rope_neox_subdim`, ...). Validate the derived rotary dimension while you
are there: `partial_rotary_factor * head_dim` must be a positive even
integer, since the NeoX sub-dimension RoPE rotates half that many pairs and
an odd value silently drops a channel.

The rest of this phase only bites on a real download:

- [ ] **Quantization widths are validated per slot and the allowed sets
      are narrow.** `validate_quant` (`crates/model-io/src/manifest.rs`)
      accepts embedding 4, attention 4, router 8, sharedExpert 4 or 8,
      routedExpert 2 or 4, and demands `affine` / bf16 scales / bf16
      biases / group size exactly 64 on every one. A family whose router
      ships INT4, or whose checkpoint uses a group size other than 64, is
      rejected at load, not at repack.
      **Source: a block-quantized slot is validated as a different shape,
      not by widening that table.** A GGUF slot carries no weight bits and
      no group size (the scale lives inside the block), so it is accepted
      as `scheme` plus `ggmlType` against `model_io::EXECUTABLE_GGUF_TYPES`.
      Two independent gates gate it and they must move together or
      `crates/runtime/tests/gguf_install_refused.rs` reddens: that one on
      the manifest's claim, and `RealForwardRunner::open` on the resident
      index's dtype tags, which believes the bytes. And what is executable
      is decided per block type, not per format: a type needs a resident
      GEMV, an embedding lookup and a routed-expert decode pair before an
      install runs, so widening the set means landing kernels rather than
      editing a list. Read the checkpoint's
      `config.json -> quantization` first (`parse_gemma4_quantization`,
      which also refuses a per-tensor group size that differs from the
      global one) and pick the `manifest_quant` probe names to match:
      the probes read layer 0, so a family whose layer 0 is not a plain
      attention layer needs a different probe (Qwen's is
      `linear_attn.in_proj_qkv`, because its layer 0 is linear and has no
      `self_attn.q_proj` at all).
- [ ] **Real checkpoints are multi-shard, and a companion tensor can live
      in a different shard than its weight.** Go through `Gemma4Shards`,
      which merges every shard's names into one registry, rather than
      resolving `.scales` / `.biases` inside the shard that held the
      weight. The `model.safetensors.index.json` weight map is the shard
      list.
- [ ] **Use the streaming writer for anything multi-GB.**
      `write_gemma4_install_streamed` computes the expert stride from
      shard headers alone, writes the resident region once, then downloads
      / writes / drops one layer's expert blobs at a time, so peak memory
      is one layer rather than the whole model. The in-memory
      `write_gemma4_install` goes through the same `StreamingGturboWriter`
      on purpose, so the two stay byte-identical; a test asserts that
      (`streamed_install_matches_in_memory_install`).
- [ ] **The routed-expert blob layout is a contract shared with the MoE
      kernels.** Nine sub-tensors per expert, `gate, gate_scales,
      gate_biases, up, ..., down, ...` back to back; one page-rounded
      (16 KiB) `expert_stride` for the WHOLE model, computed as the max
      across layers; and `down`'s blob offset must be 4-byte aligned
      because the phase-2 row helper reads its weights with `uint` loads
      (the writer checks and errors). The manifest's `expertStride` is
      separately validated as 4 KiB-aligned at load.
- [ ] Add the network repack test next to
      `crates/repack/tests/gemma4_checkpoint_network.rs`: `#[ignore]`d,
      `--release`, downloads the real checkpoint. It is not part of the
      handoff gate, but it is the only thing that proves the shard walk.

Gate: `cargo test -p turbospark-repack` passes, and the synthetic install
opens through `RealForwardRunner::open` without touching decode.

---

## Phase 2 - Kernels, each parity-tested in isolation

- [ ] For each new kernel, vendor the MSL under `crates/gpu/src/shaders/`
      byte-for-byte (`diff` it against the Swift original and keep the
      diff empty; a port-local edit belongs in a separate, commented
      kernel) and write a CPU reference in
      `crates/compute` if one does not exist. A kernel with no reference is
      a kernel you cannot verify; `logit.metal`'s `sample` was descoped for
      exactly this reason.
- [ ] Add a parity test in `crates/gpu/tests/` against that reference on
      real hardware. Cover the saturating / wrapping / edge inputs, not
      just the middle of the range.
- [ ] **Mutation-check the parity test before trusting it.** Break the
      kernel in the way you most fear (a sign, a stride, a hoisted scale)
      and confirm the suite goes red, then restore. A parity test written
      from the same mental model as the kernel can agree with it while
      both are wrong, and a passing test that cannot fail is worse than
      no test because it is believed. Done for
      `dequant_q8_0_gemv_simd`: flipping its signed read to unsigned fails
      all four cases.
- [ ] Port-local kernels (no Swift original) are allowed - `scalar_mul_fp16`
      and `logit_softcap_fp16` (both in `crates/gpu/src/shaders/utility.metal`)
      are two - but the shader comment must say
      so and say why the fused upstream form does not fit. A whole
      port-local file is also fine when there is no upstream at all to
      diff against: `shaders/dequant_q8_0.metal` is the precedent, since
      Swift has no GGUF intake. Say that in the header, and name the CPU
      reference that is then the kernel's only contract.
- [ ] **Function constants are part of the pipeline cache key**
      (`MetalContext::pipeline`, `crates/gpu/src/context.rs`). If you
      specialize a value into a pipeline, its bytes must go into
      `constants_key`, or the first dispatch's specialization gets reused
      for every later one. `encode_attention_decode`
      (`crates/gpu/src/attention_decode.rs`) keys on both `scale`
      and `ring_capacity`.
- [ ] Watch for constants the shader checks unconditionally (no
      `is_function_constant_defined` gate). Specializing those with a dummy
      value silently overrides the runtime buffer argument.
- [ ] **A kernel that calls a helper defined in another shader file needs
      the two concatenated.** The Swift build links every module into one
      library; this port compiles one library per file. The fix is a single
      `concat!(include_str!(a), "\n", include_str!(b))` constant --
      `crates/gpu/src/gdn.rs`'s `SOURCE` is the example -- which keeps one
      stable address for the address-keyed pipeline cache. Cost: the
      helper's own kernels compile twice. Prove the fused path is
      bit-identical to the separate one before relying on it.
- [ ] **A fixed threadgroup size can be a correctness contract, not a
      tuning knob.** `gdn_qk_norm` and `gdn_gated_norm` reduce their SIMD
      partials with a hardcoded `for (i = 0; i < 4; ++i)`: at anything but
      128 threads they sum uninitialized slots or drop work, with no crash
      and no compile error. Grep a new kernel for hardcoded partial-array
      bounds and register tiles (`float s[8]`) before choosing a dispatch
      shape, and turn what you find into `validate()` preconditions.
- [ ] Two test shapes beyond CPU parity, both of which caught real classes
      of bug in this port: **fused-equals-separate** (a fused multi-way
      GEMV must be byte-identical to the dispatches it replaces, since
      greedy output depends on it) and **chunk-equals-N-steps** (a prefill
      kernel over `T` rows must match `T` sequential decode steps,
      including the carried state and the `T < history` tail path). See
      `crates/gpu/tests/gdn_parity.rs`.

Gate: `cargo test -p turbospark-gpu` passes on the Metal device.

---

## Phase 3 - The decode flow, greedy first

- [ ] Implement the layer loop against the Phase 0 pseudocode. Keep the
      pseudocode in the module header and keep it accurate.
- [ ] Select the flow from `ArchConfig.family`, NOT from tensor naming.
      Every real family so far carries
      `language_model.model.embed_tokens.weight`, so a naming probe can
      only tell a real install from a synthetic short-name one. See
      `open_inner` in `crates/runtime/src/real_forward.rs`.
- [ ] If the family has recurrent per-layer state (a linear-attention
      layer's delta-rule `S` and conv tail), `reset()` has to rewind it
      too. Rewinding only the KV cache leaks the previous generation's
      context into every such layer, and the symptom is invisible: output
      stays finite and deterministic, just wrong.
- [ ] Size the KV cache from the layer mask. **Enable the SWA ring**
      (`KvCacheManager::new`'s `fp16_ring_enabled`,
      `crates/gpu/src/kv_cache.rs`) whenever the mask has
      sliding-window layers, size them
      `min(max_context, sliding_window + prefill_chunk)` (the chunk
      headroom constant is `MAX_PREFILL_CHUNK_TOKENS` in
      `crates/runtime/src/real_forward.rs`), and pass
      `ring_capacity(layer)` into `encode_attention_decode`. A comment
      claiming "all-full-attention, no ring needed" is how 600 MiB of KV
      got allocated for nothing; derive the flag from the mask, not from
      prose. The capacity must also reach the pipeline-cache constants
      key, or ring dispatches reuse the linear pipeline.
- [ ] Wrap the per-token entry point in `gpu::autorelease_pool`
      (`crates/gpu/src/context.rs`). Not
      optional - see Gotcha 17.
- [ ] Preallocate all activation scratch at open. Assert the hot path
      allocates no Metal buffers (`gpu_buffer_allocations()` flat across
      tokens); copy `decode_hot_path_allocates_no_gpu_buffers` in
      `crates/runtime/tests/real_forward_gemma4.rs`.
- [ ] `produce_prefill` skips the head but must advance every other
      side effect, recurrent state included. Guard it by asserting that
      prefill-then-decode lands on the same logits as all-`produce`
      (`prefill_then_decode_matches_all_produce`).
- [ ] **On a synthetic fixture, output-only assertions have no teeth.**
      Untrained weights make "it decoded finite deterministic tokens" true
      of a block that silently produced zeros. Assert on the block's own
      state instead: `gdn_state_abs_max` exists purely so a test can say
      the recurrence ran. Same reasoning as Gotcha 12's near-uniform
      routers.

Gate: greedy generation on the real checkpoint, with the chat template
applied, produces coherent text for 400 tokens. If it does not, the bug is
in the math, and nothing after this phase is worth debugging.

A synthetic install does NOT clear this gate. It proves the plumbing
(shapes, bindings, state carry, stop handling) and nothing about the math,
because there is no correct output to compare against. Landing the flow
against a synthetic fixture first is fine and is what Qwen 3.6 did; just
record that the real-checkpoint gate is still owed.

---

## Phase 4 - The output head and the sampler boundary

This phase gets its own section because it is where a model that looks
perfect under greedy decoding is quietly broken.

- [ ] **`produce` writes logits. The sampler softmaxes. Once.**
      `selection::select` (`crates/selection/src/choose.rs`) normalizes
      whatever it receives. A head that also
      normalizes yields `softmax(softmax(z))`, which over a large vocab is
      nearly uniform - and because softmax is monotone, ranking survives,
      so greedy output is *byte-identical to correct*. Only sampling
      exposes it, as fluent text that derails into word salad after a few
      dozen tokens, with stray unused/foreign tokens sprinkled early.
      Guards to copy: the softcap-bound assertion in
      `crates/runtime/tests/real_forward_gemma4.rs` and the
      does-not-normalize assertion in `crates/gpu/tests/utility_and_pass.rs`.
      A family with NO softcap has no bound to assert, so use the shape of
      the distribution instead: real logits are not all non-negative and do
      not sum to 1 (`crates/runtime/tests/real_forward_qwen.rs`).
- [ ] Apply the logit softcap in the head if the family has one (that is
      what HF's `*ForCausalLM.forward` returns), and stop there. If you
      need a fused cap+softmax kernel for a GPU sampler later, add it then.
- [ ] Check the softcap survives FP16 before trusting it. Dump the top-8
      raw logits for a few real tokens: if they sit deep in tanh
      saturation, many will round to the same FP16 value and ties will
      collapse to the lowest index. Gemma 4's peak around 34-38 caps to
      ~24.5, comfortably clear.
- [ ] Resolve the full EOS set from `generation_config.json` (it is a
      list), and confirm generation actually stops. `stop reason
      MaxTokens` on a short question is a symptom, not a setting.

Gate: sampled generation at the CLI defaults stays coherent for 400 tokens,
and a short question terminates with `EndOfTurn`. Run this even when greedy
already looks perfect - especially then.

---

## Phase 5 - Memory

- [ ] Write the static accounting down first: resident weights + KV +
      recurrent per-layer state + expert slot capacity + process baseline.
      Compute each from shapes, not from a measurement. The state term is
      easy to forget because it is invisible at small shapes and fixed at
      large ones: Qwen 3.6 35B is 2 MiB of delta-rule state x 30 linear
      layers = 60 MiB, plus 48 KiB of conv tail each, none of it growing
      with context. Layers that carry it carry NO KV rows in exchange.
- [ ] The resident weight mapping counts in `phys_footprint`. A plain
      read-only `mmap` would not, but `newBufferWithBytesNoCopy` pins it.
- [ ] Give the family its OWN oracle target next to
      `crates/bench/tests/memory_oracle.rs` and `qwen36_memory_oracle.rs`
      (`#[ignore]`d, gated on its own `TURBOSPARK_<FAMILY>_INSTALL_DIR` env
      var; the mach sampler is `crates/bench/src/memory.rs`). A separate
      test target, not a second `#[test]` in an existing one: the
      footprint assertion is a whole-session peak, and two families with
      different ceilings cannot share one process. What it asserts, per
      its per-chip baseline rows (each labelled with a `source`: a published
      Swift number or this port's own past measurement):
      - session peak `phys_footprint` <= the row's ceiling (~5% headroom
        already baked into Swift-derived rows). Add a row for the new
        model/chip and sanity-check it against your written accounting;
      - every protocol case stops `endOfTurn`;
      - decode tok/s >= the row's floor, where a row exists;
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

## Phase 6 - Quality

Coherence judged by eye is the gate through Phase 3, and it stops being
enough here: it cannot see a few percent of drift, which is exactly what a
quantization change looks like when it is subtly wrong rather than broken.

- [ ] Give the family its own quality-gate target next to
      `crates/bench/tests/quality_gate.rs`, sharing `quality_common`.
      Separate target, same one-model-per-process rule as the oracle.
- [ ] **Score only assistant-position tokens.** An instruction-tuned
      checkpoint is never trained to predict the prompt, so teacher-forcing
      prompt text measures nothing: on Gemma 4 it read 15.3 nats against a
      uniform-distribution bound of 12.5, worse than guessing, while
      assistant-side tokens in the same sequence scored 0.000. The corpus is
      a fixed reference answer placed in the assistant slot.
- [ ] Expect the perplexity to be not comparable to the other families'.
      Gemma's template opens a `<|channel>thought` block before the
      assistant slot and Qwen's does not, which is most of 37.31 against
      6.25 on the same passage. Each row is a sentinel against its own past.
- [ ] Freeze a greedy and a sampled digest.
- [ ] **Dispatch a layer's routed slots in the router's ranking, and assert
      the 8-slot digest equals the 16-slot one.** Phase 2 reduces in slot
      order and FP addition is not associative, so the slot order is the
      summation order: order it by anything the cache can reach and the
      same prompt decodes to different text run to run (AGENTS.md Gotcha
      27, which cost a Gemma-flow bug that survived two months because no
      test ran one generation twice on one warm runner).

Gate: the gate passes twice from two fresh processes, agreeing on every
digit and every hex character. If it does not, generation is not
deterministic and no golden digest can hold, which is a bigger problem
than whatever you were about to freeze.

---

## Phase 7 - Make it reachable from the CLI

Everything up to here is provable with an env var pointing at a directory
you built by hand. **A family that stops at Phase 6 exists for you and for
nobody else**: `turbospark-model probe` refuses it, `pull` will not install
it, and the only way to get one is to run an `#[ignore]`d test with the
right environment variable set. This phase is what turns that into
`turbospark-model pull <alias>`.

Which of the items below apply depends on the SOURCE, and the split is
sharp:

- **GGUF costs nothing here.** Both halves are family-agnostic:
  `probe/gguf.rs` resolves the family through `repack::gguf_arch_support`
  and checks block types against `model_io::EXECUTABLE_GGUF_TYPES`, and
  `install.rs::stream_gguf` calls the one `write_gguf_install_streamed`
  whatever family came back. The `arch_registry.rs` row Phase 1 already
  wrote is the whole wiring. Skip to the catalog row.
- **A safetensors / MLX family has three `match family` sites**, and they
  are the same shape as `crates/invocation`'s two parser matches
  (AGENTS.md Gotcha 14): every one ends in an `other =>` arm that returns
  an error, so a missing arm is a refusal at runtime rather than a compile
  failure.

- [ ] **`crates/catalog/src/probe/safetensors.rs::evaluate_config`** - the
      parser behind the verdict. Until the family has an arm here, `probe`
      answers `REFUSED` with "whose safetensors intake is not wired here",
      and `pull` refuses before downloading anything, which is the correct
      behavior and is indistinguishable from the family not existing.
- [ ] **`crates/catalog/src/install.rs::stream_mlx`** - the same parse
      again at install time.
- [ ] **`crates/catalog/src/install.rs`'s writer match** - the
      `write_<family>_install_streamed` call. Miss this one and the failure
      lands in the worst place of the three: the probe says `RUNNABLE`, the
      sidecars fetch and verify, and it dies at the top of the stream.
- [ ] **A new affine width is a fourth site**, and it is a conjunction
      rather than a list: `repack::is_supported_affine_shape(bits, group)`
      checks the `(bits, group_size)` pair, because the cross-products are
      combinations no published file has. The probe reports the pair it
      found either way, so a refusal here names the shape rather than
      saying "unsupported quantization".

Then the catalog row, which is the part with an admission rule rather than
a code change:

- [ ] **A row exists only if that exact repository and revision were
      streamed and run here** - the same rule `arch_registry.rs` states for
      its architecture strings. A model you expect to work is not a row.
      Install it with `pull --repo ... --alias ...` first, generate with
      it, then add the row to `crates/catalog/src/models.json`.
- [ ] Pin a commit sha where the publisher offers one. Where they do not
      (every GGUF publisher here), the row floats at `main` and
      `download_bytes` is the only fingerprint that would notice a
      re-upload, so take that figure from the network guard rather than
      from a listing page. The guard's tolerance is 2%: an earlier draft
      carried round numbers at 10%, under which `mixtral`'s recorded
      "26 GB" sat 9.4% from its real size and would have absorbed an
      entire re-quantization.
- [ ] Set `status` honestly. `verified` means a frozen quality-gate or
      memory-oracle row in `BENCHMARKS.md` asserts something about that
      exact artifact - i.e. Phase 5 and Phase 6 landed for it. `runs` means
      it generated coherent text here and nothing is frozen. `caveat` means
      it installs and runs and the notes disqualify it, and is worth using:
      `mixtral` is a row precisely because "installs, decodes, wants
      54.5 GiB of pinned slot cache" is invisible any other way.
- [ ] `install_bytes` feeds the free-space check, so keep it generous
      rather than exact. The honest post-install figure is recorded from
      `directory_bytes` at install time.
- [ ] Name the sidecars. **A GGUF row almost always needs a
      `sidecar_repo`** pointing at the checkpoint the GGUF was converted
      from: a GGUF carries its tokenizer as llama.cpp metadata and this
      port loads an HF `tokenizer.json`, and nothing about the weights repo
      says which one that is. Read the sidecar repo's own file list rather
      than copying a list from a sibling checkpoint of the same family -
      `Qwen3.8-27B` ships its merges inside `tokenizer.json` where
      `Bonsai-27B` ships `merges.txt`, and the copied list 404s.

**Model selection needs nothing family-specific, and that is worth knowing
so you do not go looking.** `catalog::resolve_model_arg` maps a `--model`
string to a path with no reference to the family, and `open_session`
(`crates/cli/src/generate.rs`) then goes through `peek_manifest_arch`,
which Phase 1 already extended. An existing directory always wins over an
alias, deliberately: a bare name that silently preferred an alias would run
a different model than the one on the command line, fluently and with no
error. So the moment the row exists, `--model <alias>` works for
`turbospark-check` and `turbospark-server` alike.

Gate: on a machine with nothing pre-installed,
`turbospark-model probe <repo>` prints `RUNNABLE`,
`turbospark-model pull <alias>` installs, and
`turbospark-check --model <alias> --messages-file p.json` generates
coherent text. Then `cargo test -p turbospark-catalog` offline, and the rot
guard (`make catalog-guard`, ~26 s, downloads nothing) with its published
byte figure pasted back into the row.

---

## Phase 8 - Write it down

- [ ] `AGENTS.md`: a Gotcha for anything a reader would get wrong twice.
- [ ] `docs/MODELS.md` and the README's model tables: a family nobody can
      find is a family nobody uses. The catalog row from Phase 7 is what
      `turbospark-model list` prints; these are where a reader learns the
      command exists.
- [ ] `DEVIATIONS.md`: every deliberate divergence from the reference, with
      the reason. "Swift fuses cap+softmax because it samples on the GPU"
      is the shape - what they do, what we do, why the difference is
      correct.
- [ ] Module headers: the layer-flow pseudocode, the memory model, and the
      output contract. Anyone debugging at 3am reads these first.
- [ ] Update this file if a new trap cost you more than an hour.
- [ ] **Record what you deliberately did NOT do, with the ceiling.** Land
      the smallest thing that clears each gate and write the rest down;
      half-wiring a throughput optimization is worse than not having it,
      because the next reader cannot tell whether it is finished. Qwen 3.6
      shipped with none of the three command-buffer overlap seams the Gemma
      path carries, its GDN prefill kernels parity-tested but unwired, and
      its input-projection function constants unspecialized -- each one a
      DEVIATIONS.md entry naming the measured or estimated ceiling, so the
      next session can rank them instead of rediscovering them.

---

## Quick triage table

| Symptom | Look here first |
|---|---|
| Greedy fine, sampled degenerates | Double softmax in the head (Gotcha 16) |
| `probe` says REFUSED on a family that decodes fine from a hand-built install | Phase 7: `probe/safetensors.rs::evaluate_config` has no arm for it. GGUF needs no arm at all, only the `arch_registry.rs` row |
| `probe` says RUNNABLE, sidecars verify, then it dies at the top of the stream | Phase 7: the writer match in `install.rs` was missed while the two parse sites were done |
| `pull` 404s on a sidecar after a long stream | It cannot: sidecars are fetched and verified FIRST (Gotcha 47). If it happened, the sidecar list is being read somewhere other than `InstallPlan` |
| `--model <alias>` runs a different model than expected | A directory of that name exists in the cwd and wins over the alias, by design. `turbospark-model path <alias>` prints the resolved directory |
| Babble from token 1 on any prompt | Chat template not applied |
| Coherent then collapses at a fixed position | KV ring capacity vs window |
| Never emits EOS | EOS set not resolved from `generation_config.json` |
| Memory grows linearly with tokens | Missing autorelease pool (Gotcha 17) |
| Memory 3-4x the reference at open | SWA layers sized at `max_context` |
| Output changed after a kernel tweak | Function constant missing from the pipeline cache key |
| Babble on an instruction-tuned model with `--prompt` | Not a decode bug: `--prompt` does no templating; use `--messages-file` or `--chat` |
| Footprint explodes, output correct | Routed-expert marker unrecognized: every expert became a resident tensor (Gotcha 26) |
| `open()` refuses with "top_k N exceeds the M-slot MoE kernels" | The checkpoint's `top_k_experts` exceeds `gpu::MAX_STREAMED_EXPERTS`, the vendored INT4-affine decode kernel's fixed reduce width -- a kernel capacity limit, not a bug (Phase 0's "top_k_experts against the decode kernel" bullet, `crates/gpu/CLAUDE.md` Gotcha 13). Widen by sizing the dispatch and reduce loop to the checkpoint's OWN `top_k`, never by dispatching every family at a new fixed width -- that doubles phase-2 cost for every pre-existing family whose `top_k` used to equal the old ceiling exactly |
| Manifest never validates, several extension fields mismatch at once | They were omitted and resolved against the GEMMA baseline (Gotcha 24); or a float field is not a binary fraction |
| Second generation differs from the first | `reset()` rewound the KV cache but not the recurrent state |
| Output differs between two `--expert-cache-slots` values | A BUG since 2026-08-08: the flow is dispatching routed slots in an order the expert cache can reach, so phase 2's reduce order follows cache state (AGENTS.md Gotcha 27). Dispatch in router rank |
| Same prompt decodes differently on a second warm run | Same cause as the row above, seen from the other side. `crates/bench/tests/gguf_nondeterminism_probe.rs` is the check |
| Loads, decodes, never errors, and the text is gibberish | A SOURCE CONVENTION, not a kernel: some tensor means something else in this checkpoint's producer. Correlate every resident tensor against a known-good install of the same model, and enumerate by the DIMENSION the difference indexes rather than by the tensors easiest to compare (AGENTS.md Gotcha 33) |
| Fixed the convention, still gibberish | The fix was scoped to the tensors your probe could reach. A BF16 probe cannot see quantized tensors on the same axis; dequantize a representative row per head and correlate (Gotcha 33) |
| Perplexity worse than a uniform distribution | Scoring prompt-position tokens on an instruction-tuned checkpoint (Phase 6) |
| A whole layer kind seems to contribute nothing | Untrained fixture: assert on the block's state, not its output |
| A perturbation-style fixture file passes every case but a real mutation survives | EVERY test in it is SELF-RELATIVE: each rebuilds its baseline inside the mutated binary, so a change to the MATH leaves them all green. Add ONE frozen digest over a deterministic synthetic install's logits -- it is the only assertion that compares against something computed before the mutation. Measured: six mutations, one reddened before, five after (AGENTS.md Gotcha 51) |
| The tokenizer refuses to load with "missing required special token" | The DIALECT PROBE resolved a family this checkpoint is not, and the resolver then required a token it lacks. Check whether the probe tests FEWER tokens than its resolver requires; a probe must not be able to pass where its own resolver will fail (Gotcha 52) |
| The whole chat template fails to parse with "unexpected identifier, expected `,`" | minijinja rejects a conditional expression as a keyword argument (`f(k=a if c else d)`), which Jinja2 accepts. `jinja_chat_template.rs::parenthesize_conditional_kwargs` rewrites it; the shim is meant to be deleted when a stable minijinja handles it (Gotcha 53) |
| A frozen row's generation stops on `maxTokens` where the prompt clearly fits | The model REASONS before answering and the shared 1,024 budget is too small. Measure the completion, not just the prompt: `protocol_parameters` needs a per-family budget, as gpt-oss and `muse_glimmer` both do |
| Throughput moved after a decode change | `MFERENCE_PHASES=1` buckets + GPU busy line, interleaved A/B pairs (`AGENTS.md` Gotcha 12); run-to-run spread is wider than most single effects |
