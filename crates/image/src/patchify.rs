//! Spatial and temporal patchify/unpatchify transformations and sequence assembly.
//!
//! Converts latents (C, F, H, W) to patch tokens with spatial patch size 2x2
//! and temporal patch size 1. Builds 3-axis coordinate grids (t, h, w) and
//! sequence padding to multiples of 32 (SEQ_MULTI_OF).

#![forbid(unsafe_code)]

pub const SEQ_MULTI_OF: usize = 32;
pub const DEFAULT_PATCH_SIZE: usize = 2;
pub const DEFAULT_F_PATCH_SIZE: usize = 1;
pub const LATENT_CHANNELS: usize = 16;
pub const PATCH_DIM: usize =
    DEFAULT_F_PATCH_SIZE * DEFAULT_PATCH_SIZE * DEFAULT_PATCH_SIZE * LATENT_CHANNELS; // 64

pub type GridSize = (usize, usize, usize);
pub type PatchifiedImage = (Vec<f32>, GridSize, GridSize);
pub type PaddedFeatures = (Vec<f32>, Vec<[i32; 3]>, Vec<bool>);

/// Patchify an image latent tensor: (C, F, H, W) -> (num_patches, patch_dim).
///
/// In memory, input is ordered as C x F x H x W (channel-first).
/// Output is ordered as (F_tokens * H_tokens * W_tokens) patches of
/// (pF * pH * pW * C) dimensions each.
pub fn patchify_image(
    image: &[f32],
    c: usize,
    f: usize,
    h: usize,
    w: usize,
    patch_size: usize,
    f_patch_size: usize,
) -> Result<PatchifiedImage, String> {
    if c == 0 || f == 0 || h == 0 || w == 0 {
        return Err("image dimensions must all be nonzero".to_string());
    }
    if patch_size == 0 || f_patch_size == 0 {
        return Err("patch sizes must both be nonzero".to_string());
    }
    if image.len() != c * f * h * w {
        return Err(format!(
            "image buffer length {} does not match C*F*H*W ({}*{}*{}*{} = {})",
            image.len(),
            c,
            f,
            h,
            w,
            c * f * h * w
        ));
    }
    if f % f_patch_size != 0 || h % patch_size != 0 || w % patch_size != 0 {
        return Err(format!(
            "dimensions ({f}, {h}, {w}) not divisible by patch sizes ({f_patch_size}, {patch_size}, {patch_size})"
        ));
    }

    let f_tokens = f / f_patch_size;
    let h_tokens = h / patch_size;
    let w_tokens = w / patch_size;
    let num_patches = f_tokens * h_tokens * w_tokens;
    let patch_dim = f_patch_size * patch_size * patch_size * c;

    let mut out = vec![0.0f32; num_patches * patch_dim];

    for ft in 0..f_tokens {
        for ht in 0..h_tokens {
            for wt in 0..w_tokens {
                let patch_idx = (ft * h_tokens + ht) * w_tokens + wt;
                for pf in 0..f_patch_size {
                    for ph in 0..patch_size {
                        for pw in 0..patch_size {
                            for ch in 0..c {
                                let orig_f = ft * f_patch_size + pf;
                                let orig_h = ht * patch_size + ph;
                                let orig_w = wt * patch_size + pw;
                                let in_idx = ((ch * f + orig_f) * h + orig_h) * w + orig_w;

                                let sub_idx = ((pf * patch_size + ph) * patch_size + pw) * c + ch;
                                let out_idx = patch_idx * patch_dim + sub_idx;
                                out[out_idx] = image[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((out, (f, h, w), (f_tokens, h_tokens, w_tokens)))
}

/// Unpatchify flat patches back into an image tensor: (num_patches, patch_dim) -> (C, F, H, W).
///
/// Mathematical inverse of patchify_image.
pub fn unpatchify(
    patches: &[f32],
    f_tokens: usize,
    h_tokens: usize,
    w_tokens: usize,
    f_patch_size: usize,
    patch_size: usize,
    out_channels: usize,
) -> Result<Vec<f32>, String> {
    if f_tokens == 0
        || h_tokens == 0
        || w_tokens == 0
        || f_patch_size == 0
        || patch_size == 0
        || out_channels == 0
    {
        return Err(
            "token counts, patch sizes, and output channels must all be nonzero".to_string(),
        );
    }
    let num_patches = f_tokens * h_tokens * w_tokens;
    let patch_dim = f_patch_size * patch_size * patch_size * out_channels;
    if patches.len() != num_patches * patch_dim {
        return Err(format!(
            "patches buffer length {} does not match required {} (tokens: {}x{}x{}, dim: {})",
            patches.len(),
            num_patches * patch_dim,
            f_tokens,
            h_tokens,
            w_tokens,
            patch_dim
        ));
    }

    let f = f_tokens * f_patch_size;
    let h = h_tokens * patch_size;
    let w = w_tokens * patch_size;
    let total_elements = out_channels * f * h * w;
    let mut out = vec![0.0f32; total_elements];

    for ft in 0..f_tokens {
        for ht in 0..h_tokens {
            for wt in 0..w_tokens {
                let patch_idx = (ft * h_tokens + ht) * w_tokens + wt;
                for pf in 0..f_patch_size {
                    for ph in 0..patch_size {
                        for pw in 0..patch_size {
                            for ch in 0..out_channels {
                                let orig_f = ft * f_patch_size + pf;
                                let orig_h = ht * patch_size + ph;
                                let orig_w = wt * patch_size + pw;
                                let out_idx = ((ch * f + orig_f) * h + orig_h) * w + orig_w;

                                let sub_idx =
                                    ((pf * patch_size + ph) * patch_size + pw) * out_channels + ch;
                                let in_idx = patch_idx * patch_dim + sub_idx;
                                out[out_idx] = patches[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(out)
}

/// Create a 3D coordinate grid with indexing="ij" and row-major flattening.
pub fn create_coordinate_grid(
    size: (usize, usize, usize),
    start: (usize, usize, usize),
) -> Vec<[i32; 3]> {
    let total = size.0 * size.1 * size.2;
    let mut grid = Vec::with_capacity(total);
    for t in start.0..start.0 + size.0 {
        for h in start.1..start.1 + size.1 {
            for w in start.2..start.2 + size.2 {
                grid.push([t as i32, h as i32, w as i32]);
            }
        }
    }
    grid
}

/// Pad a feature sequence to a multiple of SEQ_MULTI_OF (32),
/// assign position IDs, and build an attention mask.
///
/// In the attention mask: true means attend (valid token), false means pad.
pub fn pad_with_ids(
    feat: &[f32],
    feat_dim: usize,
    pos_grid_size: (usize, usize, usize),
    pos_start: (usize, usize, usize),
    pad_token: Option<&[f32]>,
) -> Result<PaddedFeatures, String> {
    if feat_dim == 0 {
        return Err("feature dimension must be nonzero".to_string());
    }
    if feat.is_empty() {
        return Err("cannot pad an empty feature sequence".to_string());
    }
    if feat.len() % feat_dim != 0 {
        return Err(format!(
            "feature length {} is not divisible by feature dimension {feat_dim}",
            feat.len()
        ));
    }
    let ori_len = feat.len() / feat_dim;
    let pad_len = (SEQ_MULTI_OF - (ori_len % SEQ_MULTI_OF)) % SEQ_MULTI_OF;
    let total_len = ori_len + pad_len;

    let ori_pos_ids = create_coordinate_grid(pos_grid_size, pos_start);
    if ori_pos_ids.len() != ori_len && ori_pos_ids.len() != total_len {
        return Err(format!(
            "coordinate grid size {} does not match ori_len {} or total_len {}",
            ori_pos_ids.len(),
            ori_len,
            total_len
        ));
    }

    let mut pos_ids = ori_pos_ids;
    let mut padded_feat = feat.to_vec();
    let mut mask = vec![true; total_len];

    if pad_len > 0 {
        if pos_ids.len() == ori_len {
            for _ in 0..pad_len {
                pos_ids.push([0, 0, 0]);
            }
        }
        for item in mask.iter_mut().take(total_len).skip(ori_len) {
            *item = false;
        }
        if let Some(token) = pad_token {
            if token.len() != feat_dim {
                return Err(format!(
                    "pad_token dimension {} does not match feat_dim {}",
                    token.len(),
                    feat_dim
                ));
            }
            for _ in 0..pad_len {
                padded_feat.extend_from_slice(token);
            }
        } else {
            // Repeat last token if no specific pad_token is supplied
            let last_token = &feat[(ori_len - 1) * feat_dim..ori_len * feat_dim];
            for _ in 0..pad_len {
                padded_feat.extend_from_slice(last_token);
            }
        }
    }

    Ok((padded_feat, pos_ids, mask))
}

/// Build unified sequence: concatenates image tokens and caption tokens [x, cap].
pub fn build_unified_sequence<T: Clone>(x_seq: &[T], cap_seq: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(x_seq.len() + cap_seq.len());
    out.extend_from_slice(x_seq);
    out.extend_from_slice(cap_seq);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patchify_unpatchify_roundtrip() {
        let (c, f, h, w) = (16, 1, 8, 8);
        let patch_size = 2;
        let f_patch_size = 1;

        let total = c * f * h * w;
        let original: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();

        let (patches, size, tokens) =
            patchify_image(&original, c, f, h, w, patch_size, f_patch_size).expect("patchify");

        assert_eq!(size, (1, 8, 8));
        assert_eq!(tokens, (1, 4, 4));
        assert_eq!(patches.len(), 16 * 64);

        let reconstructed = unpatchify(
            &patches,
            tokens.0,
            tokens.1,
            tokens.2,
            f_patch_size,
            patch_size,
            c,
        )
        .expect("unpatchify");

        assert_eq!(original.len(), reconstructed.len());
        for i in 0..original.len() {
            assert_eq!(original[i], reconstructed[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn test_create_coordinate_grid_ordering() {
        let grid = create_coordinate_grid((1, 2, 2), (33, 0, 0));
        assert_eq!(grid, vec![[33, 0, 0], [33, 0, 1], [33, 1, 0], [33, 1, 1],]);
    }

    #[test]
    fn test_pad_with_ids_boundary() {
        let feat = vec![1.0f32; 27 * 4];
        let (padded, ids, mask) =
            pad_with_ids(&feat, 4, (27, 1, 1), (1, 0, 0), Some(&[0.0, 0.0, 0.0, 0.0]))
                .expect("pad with ids");

        assert_eq!(padded.len(), 32 * 4);
        assert_eq!(ids.len(), 32);
        assert_eq!(mask.len(), 32);

        // First 27 are true, next 5 are false
        assert!(mask[..27].iter().all(|&m| m));
        assert!(mask[27..].iter().all(|&m| !m));

        // Padding tokens have ID [0, 0, 0]
        for id in &ids[27..] {
            assert_eq!(*id, [0, 0, 0]);
        }
    }
}
