//! The Inkling 2 DFlash2 block drafter on Metal. PORT-LOCAL.
//!
//! `incoai/Inkling-2-Mini-DFlash2` (BF16, 990 MB): 5 sliding-window layers,
//! block 8 (the anchor and 7 mask rows), two-tap dynamic convolutions around
//! every sublayer, context K/V projected from the same four target taps as
//! DSpark, the target's own embedding and LM head, and a low-rank candidate
//! selector. The reference is llama.cpp (ggml-org/llama.cpp#33073) and the
//! CPU drafter in `examples/inkling2_dflash2_accept.rs`.
//!
//! It shares [`DsparkState`] with the DSpark drafter (buffers, tap tracking,
//! context ring, rewind), so only the context write and the draft differ.
//!
//! Where it lives: `<install>/dflash2/{model.safetensors,config.json}`, or
//! the directory named by `TURBOSPARK_INKLING2_DFLASH2`.

use std::collections::HashMap;
use std::path::Path;

use foundation::LogitValue;

use super::dspark::{embed_normed, head_rows, DsparkConfig, DsparkState, Head, DRAFT_HEAD};
use super::lamq::LamqRuntime;
use super::verify::VERIFY_MAX_ROWS;
use super::DSPARK_TAP_ROWS;
use crate::families::qwen::{read_bf16_row, read_f16_rows, top_k_finite, CodebookView};
use crate::real_forward::{RealForwardError, RealForwardRunner};

/// The checkpoint's published shape. Anything else is refused: these set
/// which tensors exist and how the Metal passes are laid out.
const LAYERS: usize = 5;
const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const WINDOW: usize = 2048;
const SELECTOR_RANK: usize = 256;
const TOP_K: usize = 16;

/// The finish convolutions write FP16 divided by this, and the F32 residual
/// add multiplies it back. The reference is BF16: on the published checkpoint
/// these values reach 6e4 on a short prompt, at FP16's 65504 ceiling, and one
/// overflow turns its whole row into NaN. A power of two, so it is exact.
const CONV_SCALE: f32 = 32.0;

fn bad(m: impl std::fmt::Display) -> RealForwardError {
    RealForwardError::Unsupported(format!("Inkling DFlash2 drafter: {m}"))
}

/// What the selector and the conv passes need beyond [`DsparkConfig`].
#[derive(Debug)]
pub(crate) struct Dflash2Config {
    pub(crate) base: DsparkConfig,
    mask_token: u32,
}

/// Parses `config.json`, refusing every value this port does not implement.
fn parse_config(
    v: &serde_json::Value,
    hidden: usize,
    vocab: usize,
    unpadded_vocab: u32,
) -> Result<Dflash2Config, RealForwardError> {
    let d = &v["dflash_config"];
    let uint = |val: &serde_json::Value, k: &str| {
        val.as_u64()
            .map(|x| x as usize)
            .ok_or_else(|| bad(format!("config.json: {k} missing")))
    };
    let expect = |val: &serde_json::Value, k: &str, want: usize| {
        let got = uint(val, k)?;
        if got != want {
            return Err(bad(format!("{k} is {got}, expected {want}")));
        }
        Ok(())
    };
    if v["architectures"][0].as_str() != Some("DFlash2DraftModel") {
        return Err(bad("architectures is not DFlash2DraftModel"));
    }
    // The target's own activation is "situ", so this is read from the draft.
    if v["hidden_act"].as_str() != Some("silu") {
        return Err(bad("hidden_act is not silu"));
    }
    if v["is_causal"].as_bool() != Some(false) {
        return Err(bad("is_causal is not false"));
    }
    if v["rope_parameters"]["rope_type"].as_str() != Some("default") {
        return Err(bad("rope_parameters.rope_type is not default"));
    }
    let sliding = v["layer_types"].as_array().is_some_and(|a| {
        a.len() == LAYERS && a.iter().all(|t| t.as_str() == Some("sliding_attention"))
    });
    if !sliding {
        return Err(bad("layer_types is not five sliding_attention layers"));
    }
    expect(&v["hidden_size"], "hidden_size", hidden)?;
    expect(&v["vocab_size"], "vocab_size", vocab)?;
    expect(&v["num_hidden_layers"], "num_hidden_layers", LAYERS)?;
    expect(&v["num_attention_heads"], "num_attention_heads", HEADS)?;
    expect(&v["num_key_value_heads"], "num_key_value_heads", KV_HEADS)?;
    expect(&v["head_dim"], "head_dim", HEAD_DIM)?;
    expect(&v["sliding_window"], "sliding_window", WINDOW)?;
    expect(&v["num_target_layers"], "num_target_layers", 48)?;
    expect(&d["block_size"], "block_size", VERIFY_MAX_ROWS)?;
    expect(
        &d["conv_kernel_size"],
        "conv_kernel_size",
        gpu::DFLASH_TAPS as usize,
    )?;
    expect(
        &d["conv_group_size"],
        "conv_group_size",
        gpu::DFLASH_GROUP_SIZE as usize,
    )?;
    expect(&d["selector_rank"], "selector_rank", SELECTOR_RANK)?;
    expect(&d["selector_top_k"], "selector_top_k", TOP_K)?;
    // The cap is applied to the pair score after the correction. It is
    // monotonic, so a greedy pick is the same without it. "before_correction"
    // would change greedy picks and is not implemented.
    if d["selector_softcap_mode"].as_str() != Some("after_correction") {
        return Err(bad("selector_softcap_mode is not after_correction"));
    }
    let taps: Vec<u64> = d["target_layer_ids"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
        .unwrap_or_default();
    // Used unshifted: the same features `DSPARK_TAP_VISITS` records.
    if taps != [5, 23, 41, 48] {
        return Err(bad(format!(
            "target_layer_ids {taps:?}; the Inkling 2 flow records [5, 23, 41, 48]"
        )));
    }
    let mask_token = uint(&d["mask_token_id"], "mask_token_id")? as u32;
    if mask_token >= unpadded_vocab {
        return Err(bad(format!("mask_token_id {mask_token} outside the vocab")));
    }
    let float = |val: &serde_json::Value, k: &str| {
        val.as_f64()
            .map(|x| x as f32)
            .filter(|x| *x > 0.0)
            .ok_or_else(|| bad(format!("config.json: {k} missing")))
    };
    if hidden % gpu::DFLASH_GROUP_SIZE as usize != 0 {
        return Err(bad("hidden size is not a multiple of the conv group"));
    }
    Ok(Dflash2Config {
        base: DsparkConfig {
            h: hidden,
            layers: LAYERS,
            heads: HEADS,
            kvh: KV_HEADS,
            hd: HEAD_DIM,
            ffn: uint(&v["intermediate_size"], "intermediate_size")?,
            eps: float(&v["rms_norm_eps"], "rms_norm_eps")?,
            theta: float(&v["rope_parameters"]["rope_theta"], "rope_theta")?,
            rank: SELECTOR_RANK,
            block: VERIFY_MAX_ROWS,
            window: WINDOW,
            local: vec![true; LAYERS],
            taps: DSPARK_TAP_ROWS,
        },
        mask_token,
    })
}

/// Rows of `kernel_projection`: two sides of `taps * groups` coefficients.
fn conv_rows(h: usize) -> usize {
    2 * gpu::DFLASH_TAPS as usize * (h / gpu::DFLASH_GROUP_SIZE as usize)
}

/// Every tensor the draft reads, with its shape.
fn inventory(c: &DsparkConfig, vocab: usize) -> Vec<(String, Vec<usize>)> {
    let (h, kvd, qd) = (c.h, c.kvh * c.hd, c.heads * c.hd);
    let mut out: Vec<(String, Vec<usize>)> = vec![
        ("fc.weight".into(), vec![h, c.taps * h]),
        ("hidden_norm.weight".into(), vec![h]),
        ("norm.weight".into(), vec![h]),
        (
            "candidate_selector.hidden_projection.weight".into(),
            vec![c.rank, h],
        ),
        (
            "candidate_selector.predecessor_codebook".into(),
            vec![vocab, c.rank],
        ),
        (
            "candidate_selector.successor_codebook".into(),
            vec![vocab, c.rank],
        ),
    ];
    for l in 0..c.layers {
        let mut add = |s: &str, shape: Vec<usize>| out.push((format!("layers.{l}.{s}"), shape));
        add("input_layernorm.weight", vec![h]);
        add("post_attention_layernorm.weight", vec![h]);
        add("self_attn.q_norm.weight", vec![c.hd]);
        add("self_attn.k_norm.weight", vec![c.hd]);
        add("self_attn.q_proj.weight", vec![qd, h]);
        add("self_attn.k_proj.weight", vec![kvd, h]);
        add("self_attn.v_proj.weight", vec![kvd, h]);
        add("self_attn.o_proj.weight", vec![h, qd]);
        add("mlp.gate_proj.weight", vec![c.ffn, h]);
        add("mlp.up_proj.weight", vec![c.ffn, h]);
        add("mlp.down_proj.weight", vec![h, c.ffn]);
        for conv in ["attention_conv", "mlp_conv"] {
            add(
                &format!("{conv}.base_kernel"),
                vec![2, gpu::DFLASH_TAPS as usize, h],
            );
            add(
                &format!("{conv}.kernel_projection.weight"),
                vec![conv_rows(h), h],
            );
        }
    }
    out
}

/// The scratch only this drafter uses.
pub(crate) struct Dflash2Head {
    coeffs: gpu::MetalBuffer,
    conv_p: gpu::MetalBuffer,
    conv_f: gpu::MetalBuffer,
    fc_out: gpu::MetalBuffer,
    /// FP16 `[block - 1][vocab]`: the mask rows' logits.
    logits: gpu::MetalBuffer,
    /// FP16 `[block - 1][rank]`: the selector's hidden projection.
    hproj: gpu::MetalBuffer,
}

/// The greedy pair score over one step's candidates: `unary + sum(pred *
/// proj * succ)`, first maximum in candidate order (unary, then lower id).
fn pick(unary: &[f32], pred: &[f32], proj: &[f32], succ: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_score = f32::NEG_INFINITY;
    for (c, row) in succ.iter().enumerate() {
        let dot: f32 = (0..proj.len()).map(|r| pred[r] * proj[r] * row[r]).sum();
        if unary[c] + dot > best_score {
            best_score = unary[c] + dot;
            best = c;
        }
    }
    best
}

impl DsparkState {
    /// Loads the DFlash2 drafter into one buffer, tensor by tensor, and
    /// stores the normed mask embedding beside it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_dflash2(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &model_io::ResidentIndex,
        arch: &model_io::ArchConfig,
        dir: &Path,
        max_ctx: usize,
        lamq: Option<&LamqRuntime>,
        unpadded_vocab: u32,
        target_eps: f32,
    ) -> Result<Self, RealForwardError> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| bad(format!("read config.json: {e}")))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| bad(format!("config.json: {e}")))?;
        let (h, vocab) = (arch.hidden_size as usize, arch.vocab_size as usize);
        let cfg = parse_config(&v, h, vocab, unpadded_vocab)?;
        let st = model_io::SafetensorsFile::open(&dir.join("model.safetensors"))
            .map_err(RealForwardError::Model)?;

        let wanted = inventory(&cfg.base, vocab);
        let mut offs = HashMap::new();
        let mut end = 0usize;
        for (name, shape) in &wanted {
            let d = st
                .descriptor(name)
                .ok_or_else(|| bad(format!("{name} missing")))?;
            if d.dtype != "BF16" || &d.shape != shape {
                return Err(bad(format!(
                    "{name} is {} {:?}, expected BF16 {shape:?}",
                    d.dtype, d.shape
                )));
            }
            offs.insert(name.clone(), end as u64);
            end = (end + shape.iter().product::<usize>() * 2).next_multiple_of(256);
        }
        let extra = st.tensor_names().count() - wanted.len();
        if extra != 0 {
            return Err(bad(format!("{extra} tensors this port does not read")));
        }
        let mask_off = end;
        offs.insert("mask_embed".to_string(), mask_off as u64);
        end = (end + h * 2).next_multiple_of(256);
        let w = context.new_output_buffer(end as u64);
        for (name, _) in &wanted {
            let raw = st.raw_bytes(name).map_err(RealForwardError::Model)?;
            gpu::write_buffer_bytes(&w, offs[name] as usize, raw);
        }

        // The mask row enters as the target's normed embedding of the mask
        // token; it never changes, so compute it once.
        let emb = context.new_output_buffer((h * 2) as u64);
        let emb_n = context.new_output_buffer((h * 2) as u64);
        let pass = context.begin_pass_labeled("dflash2 mask embed");
        embed_normed(
            context,
            &pass,
            weights,
            index,
            lamq,
            cfg.mask_token,
            (&emb, &emb_n),
            h,
            vocab,
            target_eps,
        )?;
        pass.commit_and_wait();
        let bf16: Vec<u8> = read_f16_rows(&emb_n, 1, h)
            .into_iter()
            .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        gpu::write_buffer_bytes(&w, mask_off, &bf16);

        let c = &cfg.base;
        let b = c.block;
        let half = |n: usize| context.new_output_buffer((n * 2) as u64);
        let head = Head::Selector(Dflash2Head {
            coeffs: half(b * conv_rows(h)),
            conv_p: half(b * h),
            conv_f: half(b * h),
            fc_out: half(b * h),
            logits: half((b - 1) * vocab),
            hproj: half((b - 1) * c.rank),
        });
        Ok(DsparkState::new(context, cfg.base, w, offs, head, max_ctx))
    }

    /// Context K/V for `n` rows at `p` from the tap buffer: `fc`, norm, then
    /// each layer's own `k_proj` and `v_proj` (no conv on this path).
    pub(super) fn dflash2_context_rows(
        &self,
        head: &Dflash2Head,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        taps: (&gpu::MetalBuffer, u64),
        p: usize,
        n: usize,
    ) -> Result<(), RealForwardError> {
        let g = RealForwardError::Gpu;
        let c = &self.cfg;
        let (h, kvd) = (c.h, c.kvh * c.hd);
        let stride = (c.taps * h) as u32;
        gpu::encode_inkling2_bf16_gemv_rows(
            context,
            pass,
            &self.w,
            &[((taps.0, taps.1), stride)],
            &[gpu::Inkling2GemvRowsSegment {
                w_offset: self.off("fc.weight"),
                rows: h as u32,
                cols: stride,
                x: 0,
                y: (&head.fc_out, 0),
                y_stride: h as u32,
            }],
            n as u32,
        )
        .map_err(g)?;
        gpu::encode_inkling2_rms_rows(
            context,
            pass,
            gpu::Inkling2Rows::plain((&head.fc_out, 0), h as u32),
            Some(self.wv("hidden_norm.weight")),
            gpu::Inkling2Rows::plain((&self.cmix, 0), h as u32),
            h as u32,
            1,
            c.eps,
            n as u32,
        )
        .map_err(g)?;
        for l in 0..c.layers {
            let lp = |s: &str| format!("layers.{l}.{s}");
            self.dgemv(
                context,
                pass,
                (&self.cmix, h as u32),
                &[
                    (&lp("self_attn.k_proj.weight"), kvd, h, &self.ck, kvd as u32),
                    (&lp("self_attn.v_proj.weight"), kvd, h, &self.cv, kvd as u32),
                ],
                n,
            )?;
            let ck = gpu::Inkling2Rows::plain((&self.ck, 0), kvd as u32);
            gpu::encode_inkling2_rms_rows(
                context,
                pass,
                ck,
                Some(self.wv(&lp("self_attn.k_norm.weight"))),
                ck,
                c.hd as u32,
                c.kvh as u32,
                c.eps,
                n as u32,
            )
            .map_err(g)?;
            self.ctx_store(context, pass, l, p, n)?;
        }
        Ok(())
    }

    /// One side of a layer's dynamic convolution over the whole block.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        head: &Dflash2Head,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        x: &gpu::MetalBuffer,
        kernel: &str,
        out: &gpu::MetalBuffer,
        side: u32,
    ) -> Result<(), RealForwardError> {
        gpu::encode_dflash_grouped_conv(
            context,
            pass,
            (x, 0),
            (&head.coeffs, 0),
            self.wv(kernel),
            (out, 0),
            self.cfg.block as u32,
            self.cfg.h as u32,
            side,
            // Side 1 is the finish call, the one that leaves FP16's range.
            if side == 1 { CONV_SCALE.recip() } else { 1.0 },
        )
        .map_err(RealForwardError::Gpu)
    }

    /// Adds the finish convolution's output to the residual stream.
    fn add_conv(
        &self,
        head: &Dflash2Head,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
    ) -> Result<(), RealForwardError> {
        gpu::encode_dspark_residual_scaled(
            context,
            pass,
            (&self.x, 0),
            (&head.conv_f, 0),
            (self.cfg.block * self.cfg.h) as u32,
            CONV_SCALE,
        )
        .map_err(RealForwardError::Gpu)
    }
}

impl RealForwardRunner {
    /// The last DFlash2 draft's block input and its post-final-norm rows,
    /// both `[block][hidden]`: what the CPU parity test compares, short of
    /// the target head (`tests/real_forward_inkling2_dflash2_parity.rs`).
    pub fn inkling2_dflash2_block_rows(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        let d = self.real_inkling2.as_ref()?.dspark.as_ref()?;
        if !matches!(d.head, Head::Selector(_)) {
            return None;
        }
        let (b, h) = (d.cfg.block, d.cfg.h);
        let bf16 = |buf: &gpu::MetalBuffer, off: usize| -> Vec<f32> {
            gpu::read_buffer_bytes(buf, off, h * 2)
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect()
        };
        let mut input: Vec<f32> = read_f16_rows(&d.emb_n, 1, h)
            .into_iter()
            .map(|v| half::bf16::from_f32(v).to_f32())
            .collect();
        let mask = bf16(&d.w, d.off("mask_embed") as usize);
        for _ in 1..b {
            input.extend_from_slice(&mask);
        }
        Some((input, read_f16_rows(&d.hid, b, h)))
    }

    /// One block draft anchored at `anchor` (the token at `base`, not yet
    /// fed): `block - 1` proposals for `base + 1 ..`. The caller has
    /// validated the anchor and the context cursor.
    pub(super) fn inkling2_dflash2_draft(
        &mut self,
        anchor: i32,
        base: usize,
        ctx_rows: usize,
    ) -> Result<Vec<i32>, RealForwardError> {
        let g = RealForwardError::Gpu;
        let (h, vocab) = (
            self.arch.hidden_size as usize,
            self.arch.vocab_size as usize,
        );
        let pass = self.context.begin_pass_labeled("dflash2 draft");
        self.dspark_write_context(&pass, ctx_rows, base)?;
        let st = self.real_inkling2.as_ref().expect("checked");
        let d = st.dspark.as_ref().expect("checked");
        let Head::Selector(head) = &d.head else {
            return Err(bad("the open drafter is not DFlash2"));
        };
        let c = d.cfg.clone();
        let (b, kvd, qd) = (c.block, c.kvh * c.hd, c.heads * c.hd);
        let cr = conv_rows(h);
        let ctx = &mut self.context;

        // Row 0 is the anchor, rows 1.. the normed mask embedding.
        embed_normed(
            ctx,
            &pass,
            &self.weights,
            &self.index,
            st.lamq.as_deref(),
            anchor as u32,
            (&d.emb, &d.emb_n),
            h,
            vocab,
            st.cfg.rms_eps,
        )?;
        gpu::encode_dspark_init_x(
            ctx,
            &pass,
            (&d.emb_n, 0),
            d.wv("mask_embed"),
            (&d.x, 0),
            h as u32,
            b as u32,
        )
        .map_err(g)?;

        for l in 0..c.layers {
            let lp = |s: &str| format!("layers.{l}.{s}");
            let norm = |ctx: &mut gpu::MetalContext, w: &str| {
                gpu::encode_dspark_rms_f32(
                    ctx,
                    &pass,
                    (&d.x, 0),
                    d.wv(&lp(w)),
                    (&d.xn, 0),
                    h as u32,
                    c.eps,
                    b as u32,
                )
                .map_err(g)
            };

            // Attention: the sublayer reads the side-0 convolution of the
            // normed rows and its output goes through side 1.
            let attn_kernel = lp("attention_conv.base_kernel");
            norm(ctx, "input_layernorm.weight")?;
            d.dgemv(
                ctx,
                &pass,
                (&d.xn, h as u32),
                &[(
                    &lp("attention_conv.kernel_projection.weight"),
                    cr,
                    h,
                    &head.coeffs,
                    cr as u32,
                )],
                b,
            )?;
            d.conv(head, ctx, &pass, &d.xn, &attn_kernel, &head.conv_p, 0)?;
            d.dgemv(
                ctx,
                &pass,
                (&head.conv_p, h as u32),
                &[
                    (&lp("self_attn.q_proj.weight"), qd, h, &d.q, qd as u32),
                    (&lp("self_attn.k_proj.weight"), kvd, h, &d.kb, kvd as u32),
                    (&lp("self_attn.v_proj.weight"), kvd, h, &d.vb, kvd as u32),
                ],
                b,
            )?;
            d.attention(ctx, &pass, l, base)?;
            d.dgemv(
                ctx,
                &pass,
                (&d.att, qd as u32),
                &[(&lp("self_attn.o_proj.weight"), h, qd, &d.o, h as u32)],
                b,
            )?;
            d.conv(head, ctx, &pass, &d.o, &attn_kernel, &head.conv_f, 1)?;
            d.add_conv(head, ctx, &pass)?;

            // The MLP, wrapped the same way.
            let mlp_kernel = lp("mlp_conv.base_kernel");
            norm(ctx, "post_attention_layernorm.weight")?;
            d.dgemv(
                ctx,
                &pass,
                (&d.xn, h as u32),
                &[(
                    &lp("mlp_conv.kernel_projection.weight"),
                    cr,
                    h,
                    &head.coeffs,
                    cr as u32,
                )],
                b,
            )?;
            d.conv(head, ctx, &pass, &d.xn, &mlp_kernel, &head.conv_p, 0)?;
            d.dgemv(
                ctx,
                &pass,
                (&head.conv_p, h as u32),
                &[
                    (&lp("mlp.gate_proj.weight"), c.ffn, h, &d.g, c.ffn as u32),
                    (&lp("mlp.up_proj.weight"), c.ffn, h, &d.u, c.ffn as u32),
                ],
                b,
            )?;
            gpu::encode_dspark_silu_mul(
                ctx,
                &pass,
                (&d.g, 0),
                (&d.u, 0),
                (&d.act, 0),
                (b * c.ffn) as u32,
            )
            .map_err(g)?;
            d.dgemv(
                ctx,
                &pass,
                (&d.act, c.ffn as u32),
                &[(&lp("mlp.down_proj.weight"), h, c.ffn, &d.o, h as u32)],
                b,
            )?;
            d.conv(head, ctx, &pass, &d.o, &mlp_kernel, &head.conv_f, 1)?;
            d.add_conv(head, ctx, &pass)?;
        }
        gpu::encode_dspark_rms_f32(
            ctx,
            &pass,
            (&d.x, 0),
            d.wv("norm.weight"),
            (&d.hid, 0),
            h as u32,
            c.eps,
            b as u32,
        )
        .map_err(g)?;
        // Only the mask rows propose.
        let mask_rows = (&d.hid, (h * 2) as u64, h as u32);
        head_rows(
            ctx,
            &pass,
            &self.weights,
            &self.index,
            st.lamq.as_deref(),
            DRAFT_HEAD,
            vocab,
            h,
            mask_rows,
            (&head.logits, 0, vocab as u32),
            b - 1,
        )?;
        gpu::encode_inkling2_bf16_gemv_rows(
            ctx,
            &pass,
            &d.w,
            &[((mask_rows.0, mask_rows.1), mask_rows.2)],
            &[gpu::Inkling2GemvRowsSegment {
                w_offset: d.off("candidate_selector.hidden_projection.weight"),
                rows: c.rank as u32,
                cols: h as u32,
                x: 0,
                y: (&head.hproj, 0),
                y_stride: c.rank as u32,
            }],
            (b - 1) as u32,
        )
        .map_err(g)?;
        pass.commit_and_wait();

        // The selector walks on the host, over real token ids only: the
        // head and codebooks carry padding rows past `unpadded_vocab`.
        let unpadded = st.cfg.unpadded_vocab as usize;
        let book = |name: &str| CodebookView {
            buffer: &d.w,
            offset: d.off(name),
            rows: unpadded,
            rank: c.rank,
        };
        let pred = book("candidate_selector.predecessor_codebook");
        let succ = book("candidate_selector.successor_codebook");
        let hproj = read_f16_rows(&head.hproj, b - 1, c.rank);
        let mut logits = vec![LogitValue::from_f32(0.0); unpadded];
        let mut prev = anchor;
        let mut out = Vec::with_capacity(b - 1);
        for step in 0..b - 1 {
            gpu::read_buffer_f16_into(&head.logits, step * vocab * 2, &mut logits);
            // A NaN row would otherwise walk to token 0 silently.
            let (cand, unary) = top_k_finite(&logits, TOP_K).map_err(|id| {
                bad(format!(
                    "non-finite logit (mask row {}, id {id}); an intermediate left FP16 range",
                    step + 1
                ))
            })?;
            let rows = cand
                .iter()
                .map(|&t| read_bf16_row(succ, t))
                .collect::<Result<Vec<_>, _>>()?;
            let proj = &hproj[step * c.rank..(step + 1) * c.rank];
            prev = cand[pick(&unary, &read_bf16_row(pred, prev)?, proj, &rows)];
            out.push(prev);
        }
        self.real_inkling2
            .as_mut()
            .and_then(|s| s.dspark.as_mut())
            .expect("checked")
            .ctx_rows = base;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `config.json`, trimmed to the keys the parser reads.
    fn published() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "head_dim": 128, "hidden_act": "silu", "hidden_size": 2304,
            "intermediate_size": 6912, "is_causal": false,
            "layer_types": ["sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "sliding_attention"],
            "num_attention_heads": 32, "num_hidden_layers": 5,
            "num_key_value_heads": 8, "num_target_layers": 48,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "sliding_window": 2048, "vocab_size": 201024,
            "dflash_config": {
                "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                "mask_token_id": 200056, "selector_rank": 256,
                "selector_softcap_mode": "after_correction",
                "selector_top_k": 16, "target_layer_ids": [5, 23, 41, 48]
            }
        })
    }

    fn parse(v: &serde_json::Value) -> Result<Dflash2Config, RealForwardError> {
        parse_config(v, 2304, 201024, 200058)
    }

    #[test]
    fn the_published_config_parses_with_the_drafts_own_theta() {
        let c = parse(&published()).unwrap();
        assert_eq!(c.mask_token, 200056);
        assert_eq!(c.base.theta, 10000.0);
        assert_eq!(c.base.eps, 1e-6);
        assert_eq!((c.base.heads, c.base.kvh, c.base.ffn), (32, 8, 6912));
    }

    #[test]
    fn values_this_port_does_not_implement_are_refused() {
        for (path, value) in [
            ("/hidden_act", serde_json::json!("situ")),
            ("/is_causal", serde_json::json!(true)),
            ("/hidden_size", serde_json::json!(2048)),
            ("/layer_types/0", serde_json::json!("full_attention")),
            ("/dflash_config/block_size", serde_json::json!(16)),
            ("/dflash_config/mask_token_id", serde_json::json!(200058)),
            (
                "/dflash_config/selector_softcap_mode",
                serde_json::json!("before_correction"),
            ),
            (
                "/dflash_config/target_layer_ids",
                serde_json::json!([5, 19, 33, 47]),
            ),
        ] {
            let mut v = published();
            *v.pointer_mut(path).expect("key exists") = value;
            assert!(parse(&v).is_err(), "{path} was accepted");
        }
    }

    #[test]
    fn the_inventory_matches_the_published_header() {
        let c = parse(&published()).unwrap().base;
        let inv = inventory(&c, 201024);
        // 6 top-level tensors and 15 per layer, as the checkpoint header has.
        assert_eq!(inv.len(), 6 + 5 * 15);
        let shape = |n: &str| inv.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(shape("fc.weight"), [2304, 9216]);
        assert_eq!(shape("layers.4.self_attn.q_proj.weight"), [4096, 2304]);
        assert_eq!(shape("layers.0.mlp_conv.base_kernel"), [2, 2, 2304]);
        assert_eq!(
            shape("layers.0.attention_conv.kernel_projection.weight"),
            [576, 2304]
        );
    }

    #[test]
    fn the_pick_adds_the_pair_correction_and_keeps_the_first_maximum() {
        let (pred, proj) = ([1.0, 1.0], [1.0, 2.0]);
        // Candidate 1 trails on unary logits and wins on the correction.
        let succ = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        assert_eq!(pick(&[5.0, 3.0], &pred, &proj, &succ), 1);
        // An exact tie keeps the earlier candidate.
        assert_eq!(pick(&[6.0, 3.0], &pred, &proj, &succ), 0);
    }
}
