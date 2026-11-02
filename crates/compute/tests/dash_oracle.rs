//! `compute::dash` against vectors dumped from mlx-vlm
//! (`scripts/unlimited_ocr2_dash_oracle.py`). A route error stays fluent, so
//! this is the only thing that can say the selection is upstream's.

use turbospark_compute::dash::{
    block_capacity, dash_select, entmax15, page2d_inv_freq, summarize_block, DashDecodeState,
    DashParams, DashSelection, PromptPages,
};

pub struct Params {
    pub block_size: usize,
    pub local_blocks: usize,
    pub index_dim: usize,
    pub topk: usize,
    pub topm: usize,
    pub sigma: f32,
    pub page_mod: usize,
    pub per_page_image_tokens: usize,
    pub image_token_id: i32,
    pub page_token_id: i32,
    pub idx_scale: &'static [u32],
    pub floor: &'static [u32],
}
pub struct Entmax {
    pub x: &'static [u32],
    pub y: &'static [u32],
}
pub struct Summaries {
    pub blocks: &'static [u32],
    pub head_cls: &'static [u32],
    pub out: &'static [u32],
}
pub struct Select {
    pub q: &'static [u32],
    pub summaries: &'static [u32],
    pub capacity: usize,
    pub nblk: usize,
    pub pos: usize,
    pub candidates: &'static [u8],
    pub idx: &'static [i64],
    pub mask: &'static [u32],
}
pub struct Step {
    pub token: i32,
    pub tok: i64,
    pub page: i64,
    pub candidates: &'static [u8],
    pub cos: &'static [u32],
    pub sin: &'static [u32],
}
pub struct Pages {
    pub ids: &'static [i32],
    pub tok: &'static [i64],
    pub page: &'static [i64],
    pub query_pages: &'static [i64],
    pub blk_min: &'static [i64],
    pub blk_max: &'static [i64],
    pub dec_blk_min: &'static [i64],
    pub dec_blk_max: &'static [i64],
    pub dec_prompt_pages: i64,
    pub marker_count: i64,
    pub n_pages: i64,
    pub pf_cnt: &'static [i64],
    pub page_aware: bool,
    pub nblk: usize,
    pub prefill_candidates: &'static [u8],
    pub steps: &'static [Step],
}
pub struct Oracle {
    pub params: Params,
    pub inv_tok: &'static [u32],
    pub inv_page: &'static [u32],
    pub entmax: &'static [Entmax],
    pub summaries: Summaries,
    pub selects: &'static [Select],
    pub pages: &'static [Pages],
}

include!("generated/dash_oracle.rs");

fn floats(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|b| f32::from_bits(*b)).collect()
}

fn params() -> DashParams {
    let p = &ORACLE.params;
    DashParams {
        block_size: p.block_size,
        local_blocks: p.local_blocks,
        index_dim: p.index_dim,
        topk: p.topk,
        topm: p.topm,
        sigma: p.sigma,
        image_token_id: p.image_token_id,
        page_token_id: p.page_token_id,
        per_page_image_tokens: p.per_page_image_tokens,
        rope_page_mod: p.page_mod,
    }
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}[{i}]: got {g}, want {w}"
        );
    }
}

fn bools(flags: &[u8]) -> Vec<bool> {
    flags.iter().map(|f| *f != 0).collect()
}

#[test]
fn page2d_frequencies_match() {
    let (tok, page) = page2d_inv_freq(128, 10000.0, 256.0, 1);
    assert_close(&tok, &floats(ORACLE.inv_tok), 1e-6, "inv_tok");
    assert_close(&page, &floats(ORACLE.inv_page), 1e-6, "inv_page");
}

#[test]
fn entmax_matches_on_every_support_size() {
    for (n, case) in ORACLE.entmax.iter().enumerate() {
        let mut x = floats(case.x);
        entmax15(&mut x);
        assert_close(&x, &floats(case.y), 2e-5, &format!("entmax case {n}"));
    }
}

#[test]
fn block_summaries_match() {
    let s = &ORACLE.summaries;
    let (blocks, cls, want) = (floats(s.blocks), floats(s.head_cls), floats(s.out));
    let p = params();
    let d = p.index_dim;
    let mut out = vec![0.0; d];
    for (b, block) in blocks.chunks_exact(p.block_size * d).enumerate() {
        summarize_block(block, &cls, &mut out);
        assert_close(&out, &want[b * d..(b + 1) * d], 2e-5, "summary");
    }
}

#[test]
fn selection_is_upstreams_key_set_and_bias() {
    let p = params();
    let idx_scale = f32::from_bits(ORACLE.params.idx_scale[0]);
    let floor = f32::from_bits(ORACLE.params.floor[0]);
    let mut remote_seen = 0;
    for (n, case) in ORACLE.selects.iter().enumerate() {
        assert_eq!(block_capacity(case.nblk), case.capacity);
        let mut got = DashSelection::default();
        dash_select(
            &floats(case.q),
            &floats(case.summaries),
            &bools(case.candidates),
            case.capacity,
            case.pos,
            idx_scale,
            floor,
            &p,
            &mut got,
        );
        // Upstream's list holds masked placeholders; the attended keys are the
        // finite-mask entries. Order is only summation order, so compare sets
        // keyed by position.
        let mask = floats(case.mask);
        let mut want: Vec<(u32, f32)> = case
            .idx
            .iter()
            .zip(&mask)
            .filter(|(_, m)| m.is_finite())
            .map(|(i, m)| (*i as u32, *m))
            .collect();
        want.sort_by_key(|e| e.0);
        let mut have: Vec<(u32, f32)> = got
            .positions
            .iter()
            .copied()
            .zip(got.bias.iter().copied())
            .collect();
        have.sort_by_key(|e| e.0);
        let wp: Vec<u32> = want.iter().map(|e| e.0).collect();
        let hp: Vec<u32> = have.iter().map(|e| e.0).collect();
        assert_eq!(hp, wp, "select case {n}: attended positions");
        let wb: Vec<f32> = want.iter().map(|e| e.1).collect();
        let hb: Vec<f32> = have.iter().map(|e| e.1).collect();
        assert_close(&hb, &wb, 2e-4, &format!("select case {n}: bias"));
        remote_seen += hb.iter().filter(|b| **b != 0.0).count();
    }
    // A key list of only local tokens would pass every position check on the
    // no-remote case, so require that real bias was compared.
    assert!(
        remote_seen > 1000,
        "only {remote_seen} biased keys compared"
    );
}

#[test]
fn prompt_pages_match() {
    let p = params();
    for (n, case) in ORACLE.pages.iter().enumerate() {
        let got = PromptPages::new(case.ids, &p);
        assert_eq!(got.query_pages, case.query_pages, "case {n}: query pages");
        assert_eq!(got.blk_min, case.blk_min, "case {n}: blk_min");
        assert_eq!(got.blk_max, case.blk_max, "case {n}: blk_max");
        assert_eq!(got.dec_blk_min, case.dec_blk_min, "case {n}: dec_blk_min");
        assert_eq!(got.dec_blk_max, case.dec_blk_max, "case {n}: dec_blk_max");
        assert_eq!(got.dec_prompt_pages, case.dec_prompt_pages, "case {n}");
        assert_eq!(got.marker_count, case.marker_count, "case {n}");
        assert_eq!(got.page_aware, case.page_aware, "case {n}");
        assert_eq!(got.n_pages, case.n_pages, "case {n}");
        assert_eq!(got.pf_cnt, case.pf_cnt, "case {n}");
        match &got.coords {
            Some((tok, page)) => {
                assert_eq!(tok, case.tok, "case {n}: tok");
                assert_eq!(page, case.page, "case {n}: page");
            }
            None => assert!(case.tok.is_empty(), "case {n}: coords expected"),
        }
        if !case.prefill_candidates.is_empty() {
            let want = bools(case.prefill_candidates);
            let mut row = Vec::new();
            for (pos, expected) in want.chunks_exact(case.nblk).enumerate() {
                got.prefill_candidates(pos, case.nblk, &mut row);
                assert_eq!(&row[..], expected, "case {n}: prefill candidates at {pos}");
            }
        }
    }
}

#[test]
fn decode_state_walks_upstreams_pages() {
    let p = params();
    let (inv_tok, inv_page) = page2d_inv_freq(128, 10000.0, 256.0, 1);
    let mut angles = vec![0.0; 64];
    for (n, case) in ORACLE.pages.iter().enumerate() {
        let pages = PromptPages::new(case.ids, &p);
        let mut state = DashDecodeState::start(&pages);
        let mut offset = case.ids.len();
        let mut row = Vec::new();
        for (s, step) in case.steps.iter().enumerate() {
            let coords = state.advance(step.token, &pages, &p);
            let want = (step.tok >= 0).then_some((step.tok, step.page));
            assert_eq!(coords, want, "case {n} step {s}: coords");
            let capacity = block_capacity((offset + 1).div_ceil(p.block_size));
            state.candidates(&pages, capacity, &mut row);
            assert_eq!(row, bools(step.candidates), "case {n} step {s}: candidates");
            if let Some((tok, page)) = coords {
                turbospark_compute::dash::page2d_angles(
                    tok,
                    page,
                    p.rope_page_mod,
                    &inv_tok,
                    &inv_page,
                    &mut angles,
                );
                // Upstream's table doubles the angles for the NeoX pair.
                let (cos, sin) = (floats(step.cos), floats(step.sin));
                let c: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
                let s_: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
                assert_close(&c, &cos[..64], 1e-5, "cos");
                assert_close(&s_, &sin[..64], 1e-5, "sin");
            }
            offset += 1;
        }
    }
}
