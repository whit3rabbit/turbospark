//! Unlimited-OCR-2's decoder routing: page2d RoPE angles, page bookkeeping and
//! DASH block selection. PORT-LOCAL host arithmetic, not only a reference: the
//! decode flow calls this at the per-layer commit, so there is no router
//! kernel to disagree with it.
//!
//! Upstream is mlx-vlm `models/unlimited_ocr/dash.py`, itself parity-checked
//! against the checkpoint's `configure_dash_inference`.
//! `tests/dash_oracle.rs` pins this file to vectors dumped from it
//! (`scripts/unlimited_ocr2_dash_oracle.py`).
//!
//! A query attends a 128-token local window, unbiased, plus 12 remote 32-token
//! blocks picked by a per-layer router, each with a learned log-prior bias.
//! Remote candidates are the image blocks of the query's own page. A route
//! error stays finite and fluent: the model reads the wrong page region.
//!
//! One deliberate difference: upstream's router rounds its normed projections
//! through the checkpoint's BF16 before the rope, where [`router_index_qk`]
//! stays in FP32. Only the ranking of 16 scores and their prior read the result,
//! so it should rarely move a selection, but that is an expectation and not a
//! measurement: the real-model gate is what would show it.

const SCORE_ACT_CLAMP: f32 = 1.0e4;
const ENTMAX_NEG: f32 = -1.0e4;
/// Upstream's `_BLOCK_CAPACITY`. Part of the arithmetic, not only a compile
/// cache detail: it sizes the padded candidate row, hence `m` below.
pub const DASH_BLOCK_CAPACITY: usize = 64;

/// `sparse_attention_config`, for one layer's routing.
#[derive(Debug, Clone, PartialEq)]
pub struct DashParams {
    pub block_size: usize,
    pub local_blocks: usize,
    pub index_dim: usize,
    /// `dash_topk_blocks`: remote blocks attended.
    pub topk: usize,
    /// Scores entering the entmax prior (upstream's `DASH_INFER_TOPM`).
    pub topm: usize,
    pub sigma: f32,
    pub image_token_id: i32,
    pub page_token_id: i32,
    pub per_page_image_tokens: usize,
    /// `sparse_rope_page_mod`: the page coordinate is taken modulo this.
    pub rope_page_mod: usize,
}

impl DashParams {
    pub fn local_window(&self) -> usize {
        self.local_blocks * self.block_size
    }

    /// Keys one query can attend: every remote block plus the local window.
    pub fn max_keys(&self) -> usize {
        self.topk * self.block_size + self.local_window()
    }
}

/// Upstream's `block_capacity`: blocks rounded up to a multiple of 64.
pub fn block_capacity(blocks: usize) -> usize {
    blocks.div_ceil(DASH_BLOCK_CAPACITY) * DASH_BLOCK_CAPACITY
}

/// page2d inverse frequencies, `(token pairs, page pairs)`, in the checkpoint's
/// `resample` mode: the token pairs span the whole theta range over however
/// many pairs the page axis left them.
pub fn page2d_inv_freq(
    head_dim: usize,
    tok_theta: f32,
    page_theta: f32,
    page_pairs: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_tok = head_dim / 2 - page_pairs;
    let inv_tok = (0..n_tok)
        .map(|i| 1.0 / tok_theta.powf(i as f32 / n_tok as f32))
        .collect();
    let inv_page = (0..page_pairs)
        .map(|j| 1.0 / page_theta.powf(j as f32 / page_pairs as f32))
        .collect();
    (inv_tok, inv_page)
}

/// Per-pair page2d angles for one position into `out`, token pairs then page
/// pairs. `page` is reduced modulo `page_mod` here, as upstream's rope does.
pub fn page2d_angles(
    tok: i64,
    page: i64,
    page_mod: usize,
    inv_tok: &[f32],
    inv_page: &[f32],
    out: &mut [f32],
) {
    let page = if page_mod > 0 {
        page.rem_euclid(page_mod as i64)
    } else {
        page
    };
    let (head, tail) = out.split_at_mut(inv_tok.len());
    for (a, f) in head.iter_mut().zip(inv_tok) {
        *a = tok as f32 * f;
    }
    for (a, f) in tail.iter_mut().zip(inv_page) {
        *a = page as f32 * f;
    }
}

/// NeoX rotation of one row by per-pair angles, pairing `(i, dim/2 + i)`.
pub fn rope_rotate_row(row: &mut [f32], angles: &[f32]) {
    let half = row.len() / 2;
    for (i, a) in angles.iter().enumerate() {
        let (s, c) = a.sin_cos();
        let (x0, x1) = (row[i], row[half + i]);
        row[i] = x0 * c - x1 * s;
        row[half + i] = x0 * s + x1 * c;
    }
}

/// Page bookkeeping derived once from the whole prompt: upstream's
/// `PromptPages`. Everything is indexed by absolute position or block.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptPages {
    pub length: usize,
    pub page_aware: bool,
    /// page2d `(token-in-page, page)` per prompt position. `None` selects the
    /// 1-D yarn rope: a prompt with no page marker or no image.
    pub coords: Option<(Vec<i64>, Vec<i64>)>,
    pub n_pages: i64,
    /// Tokens each page already holds at the end of the prefill.
    pub pf_cnt: Vec<i64>,
    pub query_pages: Vec<i64>,
    pub blk_min: Vec<i64>,
    pub blk_max: Vec<i64>,
    pub dec_blk_min: Vec<i64>,
    pub dec_blk_max: Vec<i64>,
    pub dec_prompt_pages: i64,
    pub marker_count: i64,
}

fn block_page_range(pages: &[i64], block_size: usize) -> (Vec<i64>, Vec<i64>) {
    let mut lo = Vec::with_capacity(pages.len().div_ceil(block_size));
    let mut hi = Vec::with_capacity(lo.capacity());
    for block in pages.chunks(block_size) {
        let max = block.iter().copied().max().unwrap_or(-1).max(-1);
        let min = block
            .iter()
            .copied()
            .filter(|&p| p >= 0)
            .min()
            .unwrap_or(-1);
        lo.push(min);
        hi.push(max);
    }
    (lo, hi)
}

impl PromptPages {
    pub fn new(ids: &[i32], p: &DashParams) -> Self {
        let s = ids.len();
        let mut cum = Vec::with_capacity(s);
        let mut running = Vec::with_capacity(s);
        let (mut markers, mut images) = (0i64, 0i64);
        let mut last_img = None;
        for (i, &id) in ids.iter().enumerate() {
            markers += i64::from(id == p.page_token_id);
            images += i64::from(id == p.image_token_id);
            if id == p.image_token_id {
                last_img = Some(i);
            }
            cum.push(markers);
            running.push(images - 1);
        }
        let prompt_pages = last_img.map_or(0, |i| cum[i]);
        let marker_mode = prompt_pages > 0;
        let per_page = p.per_page_image_tokens.max(1) as i64;
        let in_prefill = |i: usize| last_img.is_some_and(|last| i <= last);
        let is_img = |i: usize| ids[i] == p.image_token_id;

        let image_page: Vec<i64> = (0..s)
            .map(|i| match (is_img(i), marker_mode) {
                (false, _) => -1,
                (true, true) => cum[i] - 1,
                (true, false) => running[i].div_euclid(per_page),
            })
            .collect();
        let query_pages: Vec<i64> = (0..s)
            .map(|i| {
                let raw = cum[i] - 1;
                let q = if in_prefill(i) {
                    raw
                } else {
                    raw - if marker_mode { prompt_pages } else { 0 }
                };
                q.max(-1)
            })
            .collect();
        let page_aware = last_img.is_some() && query_pages.iter().any(|&q| q >= 0);
        let (blk_min, blk_max) = block_page_range(&image_page, p.block_size);

        // Decode pages restart at 0, and a marker beats the running count.
        let dec_page: Vec<i64> = (0..s)
            .map(|i| match (is_img(i), cum[i] > 0) {
                (false, _) => -1,
                (true, true) => cum[i] - 1,
                (true, false) => running[i].div_euclid(per_page),
            })
            .collect();
        let (dec_blk_min, dec_blk_max) = block_page_range(&dec_page, p.block_size);

        let mut coords = None;
        let mut n_pages = 0;
        let mut pf_cnt = Vec::new();
        if last_img.is_some() && markers > 0 && marker_mode {
            let page: Vec<i64> = (0..s)
                .map(|i| {
                    (if in_prefill(i) {
                        cum[i]
                    } else {
                        cum[i] - prompt_pages
                    })
                    .max(0)
                })
                .collect();
            n_pages = page.iter().copied().max().unwrap_or(0) + 1;
            let mut counts = vec![0i64; n_pages as usize];
            pf_cnt = vec![0i64; n_pages as usize];
            let mut tok = Vec::with_capacity(s);
            for (i, &pg) in page.iter().enumerate() {
                tok.push(counts[pg as usize]);
                counts[pg as usize] += 1;
                if in_prefill(i) {
                    pf_cnt[pg as usize] += 1;
                }
            }
            coords = Some((tok, page));
        }

        Self {
            length: s,
            page_aware,
            coords,
            n_pages,
            pf_cnt,
            query_pages,
            blk_min,
            blk_max,
            dec_blk_min,
            dec_blk_max,
            dec_prompt_pages: prompt_pages,
            marker_count: markers,
        }
    }

    /// Same-page image blocks for a prefill query at `position`, over `nblk`
    /// blocks, into `out`. Open (all true) when nothing qualifies, and when the
    /// prompt is not page aware.
    pub fn prefill_candidates(&self, position: usize, nblk: usize, out: &mut Vec<bool>) {
        out.clear();
        out.resize(nblk, false);
        if !self.page_aware {
            out.fill(true);
            return;
        }
        let q = self.query_pages[position];
        let mut any = false;
        for (b, slot) in out.iter_mut().enumerate().take(self.blk_max.len()) {
            let same =
                q >= 0 && self.blk_min[b] >= 0 && self.blk_min[b] <= q && self.blk_max[b] >= q;
            *slot = same;
            any |= same;
        }
        if !any {
            out.fill(true);
        }
    }
}

/// The decode side's page state, advanced once per generated token: upstream's
/// `build_context` past the prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DashDecodeState {
    pub markers: i64,
    pub tok: i64,
    pub page: i64,
}

impl DashDecodeState {
    pub fn start(pages: &PromptPages) -> Self {
        let (tok, page) = pages
            .coords
            .as_ref()
            .and_then(|(t, p)| Some((*t.last()?, *p.last()?)))
            .unwrap_or((-1, -1));
        Self {
            markers: pages.marker_count,
            tok,
            page,
        }
    }

    /// Consume the token entering position `offset`, and return its page2d
    /// `(tok, page)` when the prompt has coordinates.
    pub fn advance(
        &mut self,
        token: i32,
        pages: &PromptPages,
        p: &DashParams,
    ) -> Option<(i64, i64)> {
        let is_page = token == p.page_token_id;
        self.markers += i64::from(is_page);
        pages.coords.as_ref()?;
        if is_page {
            self.page = (self.page + 1).min(pages.n_pages - 1);
            self.tok = pages.pf_cnt[self.page as usize];
        } else {
            self.tok += 1;
        }
        Some((self.tok, self.page))
    }

    /// Candidate blocks for the decode query, over `capacity` blocks. A query
    /// whose page has no image block is open, as is one before the first page.
    pub fn candidates(&self, pages: &PromptPages, capacity: usize, out: &mut Vec<bool>) {
        out.clear();
        out.resize(capacity, false);
        let qp = self.markers - 1 - pages.dec_prompt_pages;
        let qp = if qp >= 0 { qp } else { -1 };
        let mut any = false;
        for (b, slot) in out.iter_mut().enumerate().take(pages.dec_blk_max.len()) {
            let same = qp >= 0
                && pages.dec_blk_min[b] >= 0
                && pages.dec_blk_min[b] <= qp
                && pages.dec_blk_max[b] >= qp;
            *slot = same;
            any |= same;
        }
        if !any {
            out.fill(true);
        }
    }
}

/// `head_cls` attention inside one completed block of index keys,
/// `[block_size * d]`, into its one summary key.
pub fn summarize_block(block: &[f32], head_cls: &[f32], out: &mut [f32]) {
    let d = head_cls.len();
    let scale = 1.0 / (d as f32).sqrt();
    let scores: Vec<f32> = block
        .chunks_exact(d)
        .map(|row| {
            let dot: f32 = row.iter().zip(head_cls).map(|(a, b)| a * b).sum();
            (dot * scale).clamp(-SCORE_ACT_CLAMP, SCORE_ACT_CLAMP)
        })
        .collect();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f32 = weights.iter().sum();
    out.fill(0.0);
    for (row, w) in block.chunks_exact(d).zip(&weights) {
        for (o, v) in out.iter_mut().zip(row) {
            *o += (w / total) * v;
        }
    }
}

/// 1.5-entmax of a row already sorted in descending order, in place.
pub fn entmax15(x: &mut [f32]) {
    let Some(&first) = x.first() else { return };
    for v in x.iter_mut() {
        *v = (*v - first) / 2.0;
    }
    let (mut sum, mut sum_sq, mut support, mut tau_star) = (0.0f32, 0.0f32, 0usize, 0.0f32);
    let mut taus = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        let rho = (i + 1) as f32;
        sum += v;
        sum_sq += v * v;
        let mean = sum / rho;
        let ss = rho * (sum_sq / rho - mean * mean);
        let delta = (1.0 - ss) / rho;
        let tau = mean - delta.max(0.0).sqrt();
        taus.push(tau);
        if tau <= v {
            support += 1;
        }
    }
    // Upstream counts `tau <= x` over the row rather than stopping at the
    // first failure, so this is a count, not a prefix length.
    if let Some(t) = taus.get(support.max(1) - 1) {
        tau_star = *t;
    }
    for v in x.iter_mut() {
        let d = (*v - tau_star).max(0.0);
        *v = d * d;
    }
}

/// The keys one query attends: positions and each one's additive log-prior.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DashSelection {
    pub positions: Vec<u32>,
    pub bias: Vec<f32>,
}

/// Route one query at absolute position `pos` and build its key list.
///
/// `summaries` is `[n * d]`, the completed blocks. `candidates` covers
/// `capacity` blocks. `idx_scale` is `dash_scale_per_layer / sqrt(d)` and
/// `floor` the layer's prior floor. Equal to upstream's `_decode_select` plus
/// its mask: a block token inside the local window is attended there, unbiased,
/// and masked slots simply do not appear in the list.
#[allow(clippy::too_many_arguments)]
pub fn dash_select(
    query: &[f32],
    summaries: &[f32],
    candidates: &[bool],
    capacity: usize,
    pos: usize,
    idx_scale: f32,
    floor: f32,
    p: &DashParams,
    out: &mut DashSelection,
) {
    let d = query.len();
    let bs = p.block_size;
    let lw = p.local_window();
    let n = summaries.len() / d;
    let scale = 1.0 / (d as f32).sqrt();
    // A block is remote only if it ends `local_blocks` before the query's own,
    // which also means it is complete and so has a summary.
    let remote_end = ((pos / bs + 1).saturating_sub(p.local_blocks)).min(capacity);
    let cand_at = |b: usize| candidates.get(b).copied().unwrap_or(false);
    let any_cand = (0..remote_end).any(cand_at);
    let selected = |b: usize| b < remote_end && (!any_cand || cand_at(b));

    let mut scored: Vec<(f32, usize)> = (0..remote_end.min(n))
        .filter(|&b| selected(b))
        .map(|b| {
            let row = &summaries[b * d..(b + 1) * d];
            let dot: f32 = query.iter().zip(row).map(|(a, c)| a * c).sum();
            (
                (dot * scale).clamp(-SCORE_ACT_CLAMP, SCORE_ACT_CLAMP) * idx_scale,
                b,
            )
        })
        .collect();
    // Index order breaks ties, and `total_cmp` keeps the order total on a NaN.
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

    let m = p.topm.min(capacity).max(1);
    let k = p.topk.min(m).max(1);
    scored.truncate(m);
    let valid = scored.len().min(k);
    let mut prior: Vec<f32> = scored.iter().map(|s| s.0).collect();
    // Unselected blocks enter the row as the entmax floor. They sort last and
    // never reach the support, but they are part of the arithmetic.
    prior.resize(m, ENTMAX_NEG);
    entmax15(&mut prior);

    let logp: Vec<f32> = prior[..valid].iter().map(|v| v.max(floor).ln()).collect();
    let mean = logp.iter().sum::<f32>() / valid.max(1) as f32;

    out.positions.clear();
    out.bias.clear();
    for (&(_, block), lp) in scored[..valid].iter().zip(&logp) {
        let bias = (lp - mean) / p.sigma;
        for t in block * bs..(block + 1) * bs {
            if t + lw <= pos {
                out.positions.push(t as u32);
                out.bias.push(bias);
            }
        }
    }
    for t in (pos + 1).saturating_sub(lw)..=pos {
        out.positions.push(t as u32);
        out.bias.push(0.0);
    }
}

/// The router's RMS norm in FP32: Gemma-style scale `1 + weight`, with the
/// activation clamps, in place.
pub fn router_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    for v in x.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    for (v, w) in x.iter_mut().zip(weight) {
        let n = (*v * inv).clamp(-SCORE_ACT_CLAMP, SCORE_ACT_CLAMP);
        *v = (n * (1.0 + w)).clamp(-SCORE_ACT_CLAMP, SCORE_ACT_CLAMP);
    }
}

/// One matrix-vector product, `y = W x`, `W` `[rows * cols]`.
fn gemv(weight: &[f32], x: &[f32], y: &mut [f32]) {
    for (out, row) in y.iter_mut().zip(weight.chunks_exact(x.len())) {
        *out = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// Weights of one layer's router, widened to FP32.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub head_cls: Vec<f32>,
}

/// The router's index query and key for one hidden row: project, norm, rope.
/// `angles` is the same table the trunk's q and k rotate by.
pub fn router_index_qk(
    x: &[f32],
    w: &RouterWeights,
    angles: &[f32],
    eps: f32,
    q: &mut [f32],
    k: &mut [f32],
) {
    let clean: Vec<f32> = x
        .iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect();
    gemv(&w.q_proj, &clean, q);
    gemv(&w.k_proj, &clean, k);
    router_norm(q, &w.q_norm, eps);
    router_norm(k, &w.k_norm, eps);
    rope_rotate_row(q, angles);
    rope_rotate_row(k, angles);
}
