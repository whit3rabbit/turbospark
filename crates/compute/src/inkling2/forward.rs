//! CPU reference forward for Inkling 2 Mini (`inkling2_loop`), one token at a
//! time, over the checkpoint's own tensors.
//!
//! PORT-LOCAL. This is the ORACLE for the Metal flow, not a runtime path: f32
//! everywhere, weights decoded on the fly from the raw checkpoint bytes, rows
//! split over threads. The graph is transcribed from llama.cpp
//! `src/models/inkling2.cpp` and `inkling2-loop.cpp` at `29b52d398` (with
//! ggml-org/llama.cpp#32704 applied, which makes every loop pass read the
//! NVFP4 outer scales). `docs/INKLING2_PHASE0.md` lists every convention this
//! file relies on and where each was confirmed.
//!
//! Tensor names are the HF checkpoint's (`model.llm.*`). The weight source
//! reports each tensor's dtype, and every read checks it, so a name wired to
//! the wrong tensor fails loudly instead of decoding garbage.

use super::{
    attention_decode, block_attn_res_mix, dequant_mxfp8_row, dequant_nvfp4_row, l2_clip,
    log_scale_tau, loop_control, rel_bias_table, rms_norm, rms_norm_unweighted, route_sigmoid,
    short_conv, situ,
};

/// Checkpoint dtypes this forward reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F32,
    U8,
    F8E4M3,
}

/// Read access to checkpoint tensors by HF name: dtype, shape and raw
/// little-endian bytes.
pub trait WeightSource: Sync {
    fn tensor(&self, name: &str) -> Option<(Dtype, &[usize], &[u8])>;
}

/// The text model's hyperparameters. [`Config::mini`] is
/// `thinkingmachines/Inkling-2-Mini` at `ed40e7da`.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub sliding_window: usize,
    /// Physical layers with full attention; every other layer is SWA.
    pub full_layers: Vec<usize>,
    pub taps: usize,
    pub res_block: usize,
    pub top_k: usize,
    pub expert_ffn: usize,
    pub route_scale: f32,
    pub gate_cap: f32,
    pub up_cap: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub eps: f32,
    pub final_softcap: f32,
    pub core_start: usize,
    pub core_end: usize,
    pub passes: usize,
    pub gate_rank: usize,
}

impl Config {
    pub fn mini() -> Self {
        Config {
            hidden: 2304,
            layers: 24,
            vocab: 200_058,
            heads: 24,
            kv_heads: 8,
            head_dim: 128,
            d_rel: 16,
            rel_extent: 1024,
            sliding_window: 1024,
            full_layers: vec![5, 11, 17, 23],
            taps: 4,
            res_block: 6,
            top_k: 8,
            expert_ffn: 1152,
            route_scale: 8.0,
            gate_cap: 4.0,
            up_cap: 25.0,
            qk_clip: 22.627_417,
            rel_clip: 8.0,
            log_floor: 8192.0,
            log_alpha: 0.1,
            eps: 1e-6,
            final_softcap: 30.0,
            core_start: 6,
            core_end: 18,
            passes: 3,
            gate_rank: 64,
        }
    }

    fn is_full(&self, layer: usize) -> bool {
        self.full_layers.contains(&layer)
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Cache slots of the unrolled loop: the physical layers plus one copy of
    /// the core per extra pass.
    pub fn slots(&self) -> usize {
        self.layers + (self.passes - 1) * (self.core_end - self.core_start)
    }
}

/// Per-slot decode state: this slot's KV cache and its four conv tails.
struct Slot {
    k: Vec<f32>,
    v: Vec<f32>,
    tail_k: Vec<Vec<f32>>,
    tail_v: Vec<Vec<f32>>,
    tail_o: Vec<Vec<f32>>,
    tail_f: Vec<Vec<f32>>,
}

/// The reference model: weights plus decode state. [`Reference::step`]
/// consumes one token and returns its unpadded, softcapped logits.
pub struct Reference<'w, W: WeightSource> {
    w: &'w W,
    cfg: Config,
    slots: Vec<Slot>,
    pos: usize,
}

fn bf16(lo: u8, hi: u8) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([lo, hi])) << 16)
}

fn decode(dt: Dtype, bytes: &[u8]) -> Vec<f32> {
    match dt {
        Dtype::Bf16 => bytes.chunks_exact(2).map(|b| bf16(b[0], b[1])).collect(),
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        other => panic!("decode: {other:?} is not a float tensor"),
    }
}

fn dot_row(dt: Dtype, row: &[u8], x: &[f32]) -> f32 {
    match dt {
        Dtype::Bf16 => row
            .chunks_exact(2)
            .zip(x)
            .map(|(b, x)| bf16(b[0], b[1]) * x)
            .sum(),
        Dtype::F32 => row
            .chunks_exact(4)
            .zip(x)
            .map(|(b, x)| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * x)
            .sum(),
        other => panic!("dot_row: {other:?} is not a float tensor"),
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// `out[r] = f(r)` for `r in 0..n`, split across threads.
fn par_map(n: usize, f: impl Fn(usize) -> f32 + Sync) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let chunk = n.div_ceil(threads()).max(1);
    std::thread::scope(|s| {
        for (c, slice) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (i, o) in slice.iter_mut().enumerate() {
                    *o = f(c * chunk + i);
                }
            });
        }
    });
    out
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

impl<'w, W: WeightSource> Reference<'w, W> {
    pub fn new(w: &'w W, cfg: Config) -> Self {
        let rows = cfg.taps - 1;
        let slots = (0..cfg.slots())
            .map(|_| Slot {
                k: Vec::new(),
                v: Vec::new(),
                tail_k: vec![vec![0.0; cfg.kv_dim()]; rows],
                tail_v: vec![vec![0.0; cfg.kv_dim()]; rows],
                tail_o: vec![vec![0.0; cfg.hidden]; rows],
                tail_f: vec![vec![0.0; cfg.hidden]; rows],
            })
            .collect();
        Reference {
            w,
            cfg,
            slots,
            pos: 0,
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    fn get(&self, name: &str) -> (Dtype, &[usize], &[u8]) {
        self.w
            .tensor(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
    }

    /// A float tensor as f32, with its shape checked.
    fn vec(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        let (dt, s, b) = self.get(name);
        assert_eq!(s, shape, "{name} shape");
        decode(dt, b)
    }

    /// `W x` for a `[rows, cols]` float tensor.
    fn matvec(&self, name: &str, rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        let (dt, s, b) = self.get(name);
        assert_eq!(s, [rows, cols], "{name} shape");
        assert_eq!(x.len(), cols, "{name} input");
        let es = if dt == Dtype::Bf16 { 2 } else { 4 };
        par_map(rows, |r| {
            dot_row(dt, &b[r * cols * es..(r + 1) * cols * es], x)
        })
    }

    fn norm(&self, x: &[f32], name: &str) -> Vec<f32> {
        rms_norm(x, &self.vec(name, &[self.cfg.hidden]), self.cfg.eps)
    }

    /// Depth mix of `blocks` plus `cur` with the rows of a `[q, hidden]` (or
    /// `[hidden]`) F32 query tensor.
    fn mix(&self, blocks: &[Vec<f32>], cur: &[f32], query: &str) -> Vec<Vec<f32>> {
        let (dt, s, b) = self.get(query);
        let q = decode(dt, b);
        let rows = if s.len() == 2 { s[0] } else { 1 };
        let queries: Vec<Vec<f32>> = q
            .chunks_exact(self.cfg.hidden)
            .map(<[f32]>::to_vec)
            .collect();
        assert_eq!(queries.len(), rows);
        let mut states = blocks.to_vec();
        states.push(cur.to_vec());
        block_attn_res_mix(&states, &queries)
    }

    fn control(&self, cur: &[f32], msg: &[f32], ip: usize, ctrl: &str) -> Vec<f32> {
        let (h, r) = (self.cfg.hidden, self.cfg.gate_rank);
        let p = format!("model.llm.layers.{ip}.controllers.{ctrl}");
        loop_control(
            cur,
            msg,
            &self.vec(&format!("{p}.down_weight"), &[r, 2 * h]),
            &self.vec(&format!("{p}.gain_weight"), &[h, r]),
            &self.vec(&format!("{p}.forget_weight"), &[h, r]),
            r,
        )
    }

    /// The routed experts of layer `ip` on the normed input `m`.
    fn experts(&self, ip: usize, m: &[f32]) -> Vec<f32> {
        let (h, f) = (self.cfg.hidden, self.cfg.expert_ffn);
        let p = format!("model.llm.layers.{ip}.mlp");
        let n_exp = self.get(&format!("{p}.gate.weight")).1[0];
        let logits = self.matvec(&format!("{p}.gate.weight"), n_exp, h, m);
        let bias = self.vec(&format!("{p}.gate.bias"), &[n_exp]);
        let (idx, wts) = route_sigmoid(&logits, &bias, self.cfg.top_k, self.cfg.route_scale);

        let (w13_dt, w13_s, w13) = self.get(&format!("{p}.experts.w13_weight"));
        let (_, _, s13) = self.get(&format!("{p}.experts.w13_weight_scale"));
        let (w2_dt, _, w2) = self.get(&format!("{p}.experts.w2_weight"));
        let (_, _, s2) = self.get(&format!("{p}.experts.w2_weight_scale"));
        let nvfp4 = match w13_dt {
            Dtype::U8 => true,
            Dtype::F8E4M3 => false,
            other => panic!("layer {ip} experts: unexpected dtype {other:?}"),
        };
        assert_eq!(w13_s[1], 2 * f, "w13 rows");
        assert_eq!(
            w2_dt, w13_dt,
            "layer {ip}: w13 and w2 use different schemes"
        );
        let outer = |name: &str, e: usize| -> f32 {
            let amax = self.vec(&format!("{p}.experts.{name}_amax"), &[n_exp]);
            let den = self.vec(&format!("{p}.experts.{name}_outer_scale_denominator"), &[1]);
            amax[e] / den[0]
        };
        // Bytes per row of a `cols`-wide projection, and of its scales.
        let row_b = |cols: usize| if nvfp4 { cols / 2 } else { cols };
        let scale_b = |cols: usize| if nvfp4 { cols / 16 } else { cols / 32 };
        let dequant = |w: &[u8], s: &[u8], e: usize, rows: usize, r: usize, cols: usize, o: f32| {
            let wr = (e * rows + r) * row_b(cols);
            let sr = (e * rows + r) * scale_b(cols);
            let mut out = vec![0.0f32; cols];
            if nvfp4 {
                dequant_nvfp4_row(
                    &w[wr..wr + row_b(cols)],
                    &s[sr..sr + scale_b(cols)],
                    o,
                    &mut out,
                );
            } else {
                dequant_mxfp8_row(
                    &w[wr..wr + row_b(cols)],
                    &s[sr..sr + scale_b(cols)],
                    &mut out,
                );
            }
            out
        };

        let mut y = vec![0.0f32; h];
        for (&e, &wt) in idx.iter().zip(&wts) {
            let (o13, o2) = if nvfp4 {
                (outer("w13_weight", e), outer("w2_weight", e))
            } else {
                (1.0, 1.0)
            };
            // w13 rows interleave gate (even) and up (odd).
            let gu = par_map(2 * f, |r| {
                let row = dequant(w13, s13, e, 2 * f, r, h, o13);
                row.iter().zip(m).map(|(a, b)| a * b).sum()
            });
            let act: Vec<f32> = (0..f)
                .map(|k| situ(gu[2 * k], gu[2 * k + 1], self.cfg.gate_cap, self.cfg.up_cap))
                .collect();
            let out = par_map(h, |r| {
                let row = dequant(w2, s2, e, h, r, f, o2);
                row.iter().zip(&act).map(|(a, b)| a * b).sum()
            });
            for (yy, o) in y.iter_mut().zip(&out) {
                *yy += wt * o;
            }
        }
        y
    }

    /// One physical layer `ip` in cache slot `slot`. `msgs` are the four
    /// cross-pass messages (Q, K, V, FFN) on passes after the first.
    fn layer(
        &mut self,
        slot: usize,
        ip: usize,
        cur: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        msgs: Option<&[Vec<f32>]>,
    ) -> Vec<f32> {
        let c = self.cfg.clone();
        let (h, hd, kvd) = (c.hidden, c.head_dim, c.kv_dim());
        let full = c.is_full(ip);
        let p = format!("model.llm.layers.{ip}");
        let inp_sa = cur.clone();

        // Attention inputs: one mixed input for SWA, one each for Q, K, V on
        // full layers; layer 0 has no residual query.
        let mut inputs = [cur.clone(), cur.clone(), cur.clone()];
        if ip > 0 {
            let mixed = self.mix(
                blocks,
                &cur,
                &format!("model.llm.attn_res.attn_queries.{}", ip - 1),
            );
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = mixed[if full { j } else { 0 }].clone();
            }
        }
        if let Some(m) = msgs {
            for (j, ctrl) in ["attn", "k", "v"].iter().enumerate() {
                inputs[j] = self.control(&inputs[j], &m[j], ip, ctrl);
            }
        }
        if !full && msgs.is_some() {
            for inp in inputs.iter_mut() {
                *inp = self.norm(inp, &format!("{p}.attn_norm.weight"));
            }
        } else if !full {
            let n = self.norm(&inputs[0], &format!("{p}.attn_norm.weight"));
            inputs = [n.clone(), n.clone(), n];
        } else {
            for (inp, which) in inputs.iter_mut().zip(["q", "k", "v"]) {
                *inp = self.norm(inp, &format!("{p}.{which}_input_norm.weight"));
            }
        }

        let a = format!("{p}.attn");
        let q = self.matvec(&format!("{a}.wq_du.weight"), c.heads * hd, h, &inputs[0]);
        let k_pre = self.matvec(&format!("{a}.wk_dv.weight"), kvd, h, &inputs[1]);
        let v_pre = self.matvec(&format!("{a}.wv_dv.weight"), kvd, h, &inputs[2]);
        let r = self.matvec(
            &format!("{a}.wr_du.weight"),
            c.heads * c.d_rel,
            h,
            &inputs[0],
        );

        // Static conv taps as [ch, taps] with tap taps-1 on the current row.
        let conv_w = |this: &Self, which: &str| -> Vec<f32> {
            if full {
                let sw = this.vec(&format!("{a}.{which}_sconv.static_w"), &[c.taps, kvd]);
                (0..kvd * c.taps)
                    .map(|i| sw[(c.taps - 1 - i % c.taps) * kvd + i / c.taps])
                    .collect()
            } else {
                this.vec(&format!("{a}.{which}_sconv.weight"), &[kvd, 1, c.taps])
            }
        };
        let kw = conv_w(self, "k");
        let vw = conv_w(self, "v");
        let (dk, dv) = if full {
            let n = c.kv_heads * c.taps;
            (
                Some(self.matvec(&format!("{a}.k_sconv.weight_proj"), n, h, &inputs[1])),
                Some(self.matvec(&format!("{a}.v_sconv.weight_proj"), n, h, &inputs[2])),
            )
        } else {
            (None, None)
        };
        let st = &mut self.slots[slot];
        let k = short_conv(
            &[k_pre],
            &kw,
            c.taps,
            &mut st.tail_k,
            dk.as_deref().map(|d| (d, c.kv_heads)),
        )
        .remove(0);
        let v = short_conv(
            &[v_pre],
            &vw,
            c.taps,
            &mut st.tail_v,
            dv.as_deref().map(|d| (d, c.kv_heads)),
        )
        .remove(0);

        let tau = if full {
            log_scale_tau(self.pos, c.log_floor, c.log_alpha)
        } else {
            1.0
        };
        let q_scale = self.vec(&format!("{a}.q_l2_clip.scale"), &[c.heads, hd]);
        let r_scale = self.vec(&format!("{a}.rel_l2_clip.scale"), &[c.heads, c.d_rel]);
        let q: Vec<f32> = (0..c.heads)
            .flat_map(|hh| {
                l2_clip(
                    &q[hh * hd..(hh + 1) * hd],
                    Some(&q_scale[hh * hd..(hh + 1) * hd]),
                    c.qk_clip,
                )
                .into_iter()
                .map(move |x| x * tau)
            })
            .collect();
        let r: Vec<f32> = (0..c.heads)
            .flat_map(|hh| {
                let s = hh * c.d_rel..(hh + 1) * c.d_rel;
                l2_clip(&r[s.clone()], Some(&r_scale[s]), c.rel_clip)
                    .into_iter()
                    .map(move |x| x * tau)
            })
            .collect();
        let k: Vec<f32> = (0..c.kv_heads)
            .flat_map(|hh| l2_clip(&k[hh * hd..(hh + 1) * hd], None, c.qk_clip))
            .collect();
        let v: Vec<f32> = (0..c.kv_heads)
            .flat_map(|hh| rms_norm_unweighted(&v[hh * hd..(hh + 1) * hd], c.eps))
            .collect();

        let sinks = self.matvec(
            &format!("{a}.attn_sink_proj.weight"),
            c.heads,
            h,
            &inputs[0],
        );
        let proj = self.vec(
            &format!("{a}.rel_logits_proj.proj"),
            &[c.d_rel, c.rel_extent],
        );
        let gate = self.matvec(
            &format!("{a}.gate_attn.weight"),
            c.heads * hd,
            h,
            &inputs[0],
        );

        let pos = self.pos;
        let st = &mut self.slots[slot];
        st.k.extend_from_slice(&k);
        st.v.extend_from_slice(&v);
        let n_pos = st.k.len() / kvd;
        assert_eq!(n_pos, pos + 1, "slot {slot} cache length");
        let first = if full {
            0
        } else {
            (pos + 1).saturating_sub(c.sliding_window)
        };
        let table = rel_bias_table(&r, &proj, c.heads, c.d_rel, c.rel_extent);
        let mut o = attention_decode(
            &q,
            &st.k,
            &st.v,
            &table,
            &sinks,
            first,
            pos,
            c.heads,
            c.kv_heads,
            hd,
            c.rel_extent,
        );
        for (x, g) in o.iter_mut().zip(&gate) {
            *x *= 1.0 / (1.0 + (-g).exp());
        }
        let o2 = self.matvec(&format!("{a}.wo_ud.weight"), h, c.heads * hd, &o);
        let post_w = self.vec(&format!("{p}.attn_sconv.weight"), &[h, 1, c.taps]);
        let st = &mut self.slots[slot];
        let attn_out = short_conv(&[o2], &post_w, c.taps, &mut st.tail_o, None).remove(0);

        let cur = if ip % c.res_block == 0 {
            blocks.push(inp_sa.clone());
            attn_out
        } else {
            add(&inp_sa, &attn_out)
        };
        let ffn_inp = cur.clone();

        let mut m = self
            .mix(
                blocks,
                &cur,
                &format!("model.llm.attn_res.mlp_queries.{ip}"),
            )
            .remove(0);
        if let Some(msg) = msgs {
            m = self.control(&m, &msg[3], ip, "mlp");
        }
        let m = self.norm(&m, &format!("{p}.mlp_norm.weight"));
        let y = self.experts(ip, &m);
        let ffn_w = self.vec(&format!("{p}.mlp_sconv.weight"), &[h, 1, c.taps]);
        let st = &mut self.slots[slot];
        let y = short_conv(&[y], &ffn_w, c.taps, &mut st.tail_f, None).remove(0);
        add(&ffn_inp, &y)
    }

    /// Consume `token` at the next position; return its logits over the
    /// unpadded vocabulary, after the final softcap.
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        let c = self.cfg.clone();
        let h = c.hidden;
        let (dt, s, b) = self.get("model.llm.embed.weight");
        assert_eq!(s[1], h);
        let es = if dt == Dtype::Bf16 { 2 } else { 4 };
        let t = token as usize;
        let emb = decode(dt, &b[t * h * es..(t + 1) * h * es]);
        let mut cur = self.norm(&emb, "model.llm.embed_norm.weight");

        let mut blocks = Vec::new();
        for ip in 0..c.core_start {
            cur = self.layer(ip, ip, cur, &mut blocks, None);
        }
        let prelude = cur.clone();
        let prelude_blocks = blocks.clone();
        let core = c.core_end - c.core_start;
        let q_all = self.vec("model.llm.middle_cycle_cross_pass_queries", &[2 * core, h]);
        let kv_all = self.vec(
            "model.llm.middle_cycle_cross_pass_kv_queries",
            &[2 * core, h],
        );
        for pass in 0..c.passes {
            let mut bank: Option<(Vec<Vec<f32>>, Vec<f32>)> = None;
            if pass > 0 {
                bank = Some((blocks[prelude_blocks.len() + 1..].to_vec(), cur.clone()));
                cur = prelude.clone();
                blocks = prelude_blocks.clone();
            }
            for ip in c.core_start..c.core_end {
                let i = ip - c.core_start;
                let slot = if pass == 0 {
                    ip
                } else {
                    c.layers + (pass - 1) * core + i
                };
                cur = match &bank {
                    None => self.layer(slot, ip, cur, &mut blocks, None),
                    Some((bank_blocks, bank_out)) => {
                        let row = |m: &[f32], k: usize| m[k * h..(k + 1) * h].to_vec();
                        let queries = vec![
                            row(&q_all, 2 * i),
                            row(&kv_all, 2 * i),
                            row(&kv_all, 2 * i + 1),
                            row(&q_all, 2 * i + 1),
                        ];
                        let mut states = bank_blocks.clone();
                        states.push(bank_out.clone());
                        let msgs = block_attn_res_mix(&states, &queries);
                        self.layer(slot, ip, cur, &mut blocks, Some(&msgs))
                    }
                };
            }
        }
        for ip in c.core_end..c.layers {
            cur = self.layer(ip, ip, cur, &mut blocks, None);
        }

        let out = self
            .mix(&blocks, &cur, "model.llm.attn_res.output_query")
            .remove(0);
        let out = self.norm(&out, "model.llm.norm.weight");
        let (_, us, _) = self.get("model.llm.unembed.weight");
        let padded = us[0];
        let mut logits = self.matvec("model.llm.unembed.weight", padded, h, &out);
        logits.truncate(c.vocab);
        for l in logits.iter_mut() {
            *l = (*l / c.final_softcap).tanh() * c.final_softcap;
        }
        self.pos += 1;
        logits
    }
}

#[cfg(test)]
mod tests {
    //! Plumbing tests on synthetic weights: a tiny config with both expert
    //! schemes, full and SWA layers, and a two-pass loop with messages. The
    //! numerics are held by the real-checkpoint comparison with llama.cpp
    //! (`docs/INKLING2_PHASE0.md`), not here.
    use super::*;
    use std::collections::HashMap;

    struct Synth(HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>);

    impl WeightSource for Synth {
        fn tensor(&self, name: &str) -> Option<(Dtype, &[usize], &[u8])> {
            self.0
                .get(name)
                .map(|(d, s, b)| (*d, s.as_slice(), b.as_slice()))
        }
    }

    fn tiny() -> Config {
        Config {
            hidden: 64,
            layers: 6,
            vocab: 50,
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            d_rel: 4,
            rel_extent: 8,
            sliding_window: 8,
            full_layers: vec![1, 5],
            taps: 4,
            res_block: 2,
            top_k: 2,
            expert_ffn: 32,
            core_start: 2,
            core_end: 4,
            passes: 2,
            gate_rank: 8,
            ..Config::mini()
        }
    }

    struct Builder {
        map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
        seed: u32,
    }

    impl Builder {
        fn next(&mut self) -> f32 {
            self.seed = self
                .seed
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            ((self.seed >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        }
        fn bf16(&mut self, name: &str, shape: &[usize], scale: f32) {
            let n: usize = shape.iter().product();
            let bytes = (0..n)
                .flat_map(|_| {
                    let v = self.next() * scale;
                    ((v.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect();
            self.map
                .insert(name.into(), (Dtype::Bf16, shape.to_vec(), bytes));
        }
        fn f32(&mut self, name: &str, shape: &[usize], scale: f32, offset: f32) {
            let n: usize = shape.iter().product();
            let bytes = (0..n)
                .flat_map(|_| (offset + self.next() * scale).to_le_bytes())
                .collect();
            self.map
                .insert(name.into(), (Dtype::F32, shape.to_vec(), bytes));
        }
        fn bytes(&mut self, name: &str, dt: Dtype, shape: &[usize], f: impl Fn(u32) -> u8) {
            let n: usize = shape.iter().product();
            let bytes = (0..n)
                .map(|_| {
                    self.next();
                    f(self.seed >> 8)
                })
                .collect();
            self.map.insert(name.into(), (dt, shape.to_vec(), bytes));
        }
    }

    fn synth(c: &Config, mxfp8_from: usize) -> Synth {
        let mut b = Builder {
            map: HashMap::new(),
            seed: 7,
        };
        let (h, hd, kvd, e, f) = (c.hidden, c.head_dim, c.kv_dim(), 8usize, c.expert_ffn);
        let w = 1.0 / (h as f32).sqrt();
        b.bf16("model.llm.embed.weight", &[64, h], 1.0);
        b.bf16("model.llm.unembed.weight", &[64, h], w);
        for n in ["embed_norm", "norm"] {
            b.f32(&format!("model.llm.{n}.weight"), &[h], 0.1, 1.0);
        }
        b.f32("model.llm.attn_res.output_query", &[h], 0.5, 0.0);
        let core = c.core_end - c.core_start;
        b.f32(
            "model.llm.middle_cycle_cross_pass_queries",
            &[2 * core, h],
            0.5,
            0.0,
        );
        b.f32(
            "model.llm.middle_cycle_cross_pass_kv_queries",
            &[2 * core, h],
            0.5,
            0.0,
        );
        for ip in 0..c.layers {
            let full = c.is_full(ip);
            let p = format!("model.llm.layers.{ip}");
            let a = format!("{p}.attn");
            if ip > 0 {
                let rows: &[usize] = if full { &[3, h] } else { &[h] };
                b.f32(
                    &format!("model.llm.attn_res.attn_queries.{}", ip - 1),
                    rows,
                    0.5,
                    0.0,
                );
            }
            b.f32(
                &format!("model.llm.attn_res.mlp_queries.{ip}"),
                &[h],
                0.5,
                0.0,
            );
            b.bf16(&format!("{a}.wq_du.weight"), &[c.heads * hd, h], w);
            b.bf16(&format!("{a}.wk_dv.weight"), &[kvd, h], w);
            b.bf16(&format!("{a}.wv_dv.weight"), &[kvd, h], w);
            b.bf16(&format!("{a}.wr_du.weight"), &[c.heads * c.d_rel, h], w);
            b.bf16(&format!("{a}.wo_ud.weight"), &[h, c.heads * hd], w);
            b.bf16(&format!("{a}.gate_attn.weight"), &[c.heads * hd, h], w);
            b.bf16(&format!("{a}.attn_sink_proj.weight"), &[c.heads, h], w);
            b.bf16(&format!("{a}.q_l2_clip.scale"), &[c.heads, hd], 1.0);
            b.bf16(&format!("{a}.rel_l2_clip.scale"), &[c.heads, c.d_rel], 1.0);
            b.bf16(
                &format!("{a}.rel_logits_proj.proj"),
                &[c.d_rel, c.rel_extent],
                1.0,
            );
            if full {
                for x in ["k", "v"] {
                    b.bf16(&format!("{a}.{x}_sconv.static_w"), &[c.taps, kvd], 0.5);
                    b.bf16(
                        &format!("{a}.{x}_sconv.weight_proj"),
                        &[c.kv_heads * c.taps, h],
                        w,
                    );
                }
                for x in ["q", "k", "v"] {
                    b.f32(&format!("{p}.{x}_input_norm.weight"), &[h], 0.1, 1.0);
                }
            } else {
                for x in ["k", "v"] {
                    b.bf16(&format!("{a}.{x}_sconv.weight"), &[kvd, 1, c.taps], 0.5);
                }
                b.f32(&format!("{p}.attn_norm.weight"), &[h], 0.1, 1.0);
            }
            b.bf16(&format!("{p}.attn_sconv.weight"), &[h, 1, c.taps], 0.5);
            b.bf16(&format!("{p}.mlp_sconv.weight"), &[h, 1, c.taps], 0.5);
            b.f32(&format!("{p}.mlp_norm.weight"), &[h], 0.1, 1.0);
            b.bf16(&format!("{p}.mlp.gate.weight"), &[e, h], w);
            b.f32(&format!("{p}.mlp.gate.bias"), &[e], 0.1, 0.0);
            let x = format!("{p}.mlp.experts");
            if ip < mxfp8_from {
                // NVFP4: random codes, E4M3 scales near 2^-3, outer ~0.5.
                let sc = |r: u32| 0x20 | (r & 7) as u8;
                b.bytes(
                    &format!("{x}.w13_weight"),
                    Dtype::U8,
                    &[e, 2 * f, h / 2],
                    |r| r as u8,
                );
                b.bytes(
                    &format!("{x}.w13_weight_scale"),
                    Dtype::F8E4M3,
                    &[e, 2 * f, h / 16],
                    sc,
                );
                b.bytes(&format!("{x}.w2_weight"), Dtype::U8, &[e, h, f / 2], |r| {
                    r as u8
                });
                b.bytes(
                    &format!("{x}.w2_weight_scale"),
                    Dtype::F8E4M3,
                    &[e, h, f / 16],
                    sc,
                );
                for n in ["w13", "w2"] {
                    b.f32(&format!("{x}.{n}_weight_amax"), &[e], 0.2, 1.5);
                    b.f32(
                        &format!("{x}.{n}_weight_outer_scale_denominator"),
                        &[1],
                        0.0,
                        3.0,
                    );
                }
            } else {
                // MXFP8: E4M3 codes with small exponents, E8M0 near 2^-4.
                let code = |r: u32| ((r & 0x80) | (0x18 + (r >> 8) % 24)) as u8;
                b.bytes(
                    &format!("{x}.w13_weight"),
                    Dtype::F8E4M3,
                    &[e, 2 * f, h],
                    code,
                );
                b.bytes(
                    &format!("{x}.w13_weight_scale"),
                    Dtype::U8,
                    &[e, 2 * f, h / 32],
                    |_| 123,
                );
                b.bytes(&format!("{x}.w2_weight"), Dtype::F8E4M3, &[e, h, f], code);
                b.bytes(
                    &format!("{x}.w2_weight_scale"),
                    Dtype::U8,
                    &[e, h, f / 32],
                    |_| 123,
                );
            }
            if (c.core_start..c.core_end).contains(&ip) {
                for ctrl in ["attn", "k", "v", "mlp"] {
                    let q = format!("{p}.controllers.{ctrl}");
                    b.bf16(&format!("{q}.down_weight"), &[c.gate_rank, 2 * h], w);
                    b.bf16(&format!("{q}.gain_weight"), &[h, c.gate_rank], 0.1);
                    b.bf16(&format!("{q}.forget_weight"), &[h, c.gate_rank], 0.1);
                }
            }
        }
        Synth(b.map)
    }

    fn run(c: &Config, w: &Synth, tokens: &[u32]) -> Vec<Vec<f32>> {
        let mut r = Reference::new(w, c.clone());
        tokens.iter().map(|&t| r.step(t)).collect()
    }

    /// Both expert schemes, full and SWA layers, both loop passes: every
    /// tensor is read with a checked shape, and the outputs are finite,
    /// non-degenerate and reproducible.
    #[test]
    fn synthetic_forward_runs_and_is_deterministic() {
        let c = tiny();
        let w = synth(&c, 4);
        let tokens = [3, 17, 42, 3, 9, 30, 11, 5, 22, 3];
        let a = run(&c, &w, &tokens);
        let b = run(&c, &w, &tokens);
        assert_eq!(a, b);
        for (i, l) in a.iter().enumerate() {
            assert_eq!(l.len(), c.vocab);
            assert!(l.iter().all(|v| v.is_finite()), "pos {i}");
            let (lo, hi) = l
                .iter()
                .fold((f32::MAX, f32::MIN), |m, &v| (m.0.min(v), m.1.max(v)));
            assert!(hi - lo > 1e-3, "pos {i}: flat logits");
            assert!(hi <= c.final_softcap && lo >= -c.final_softcap);
        }
        // Token 3 recurs: KV, conv tails and position make each one differ.
        assert_ne!(a[0], a[3]);
        assert_ne!(a[3], a[9]);
    }

    /// The loop is wired: a third pass changes the output, so pass > 0 reads
    /// its messages and its own cache slots.
    #[test]
    fn an_extra_loop_pass_changes_the_logits() {
        let c2 = tiny();
        let c3 = Config {
            passes: 3,
            ..tiny()
        };
        let w = synth(&c2, 4);
        let tokens = [3, 17, 42];
        assert_ne!(run(&c2, &w, &tokens), run(&c3, &w, &tokens));
        assert_eq!(c3.slots(), 6 + 2 * 2);
    }
}
