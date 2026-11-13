//! Unlimited-OCR-2's page layout: mlx-vlm's `tokenize_with_images`, minus the
//! tokenizer. One page is "gundam" mode, a 1024 global view plus up to 16
//! 640 local crops. Several pages are one 1024 view each.
//!
//! Views are `[side, side, 3]` FP32 in `[-1, 1]`, channel last, which is what
//! the patch embed's weight layout reads.

use crate::resize::resize_bicubic_pil;
use crate::rounding::round_half_to_even;

pub const UOCR2_BASE_SIZE: usize = 1024;
pub const UOCR2_CROP_SIZE: usize = 640;
/// The checkpoint's `processor_config.max_crops`.
pub const UOCR2_MAX_CROPS: usize = 16;
const PATCH: usize = 16;
const DOWNSAMPLE: usize = 4;
/// `int(0.5 * 255)`: upstream pads with its normalization mean.
const PAD_GRAY: u8 = 127;

/// Rows per side of a view's grid: 16 at 1024, 10 at 640.
pub fn uocr2_grid(side: usize) -> usize {
    (side / PATCH).div_ceil(DOWNSAMPLE)
}

/// One page's views.
#[derive(Debug, Clone, PartialEq)]
pub struct Uocr2Page {
    pub base: Vec<f32>,
    pub crops: Vec<Vec<f32>>,
    /// `(columns, rows)` of crops; `(1, 1)` means no crops.
    pub grid: (usize, usize),
}

impl Uocr2Page {
    /// Image token positions this page takes in the prompt.
    pub fn image_tokens(&self) -> usize {
        let base = uocr2_grid(UOCR2_BASE_SIZE);
        let mut n = (base + 1) * base + 1;
        if !self.crops.is_empty() {
            let q = uocr2_grid(UOCR2_CROP_SIZE);
            n += (q * self.grid.0 + 1) * (q * self.grid.1);
        }
        n
    }
}

fn normalize(rgb: &[u8]) -> Vec<f32> {
    rgb.iter().map(|&b| f32::from(b) / 127.5 - 1.0).collect()
}

/// PIL `ImageOps.pad` with centering 0.5, into `size`.
fn pad_into(rgb: &[u8], width: usize, height: usize, size: usize) -> Vec<u8> {
    let round = |x: f64| round_half_to_even(x) as usize;
    let (w, h, x0, y0) = if width == height {
        (size, size, 0, 0)
    } else if width > height {
        let h = round(height as f64 / width as f64 * size as f64).max(1);
        (size, h, 0, round((size - h) as f64 * 0.5))
    } else {
        let w = round(width as f64 / height as f64 * size as f64).max(1);
        (w, size, round((size - w) as f64 * 0.5), 0)
    };
    let resized = resize_bicubic_pil(rgb, width, height, w, h);
    let mut out = vec![PAD_GRAY; size * size * 3];
    for y in 0..h {
        let src = y * w * 3;
        let dst = ((y0 + y) * size + x0) * 3;
        out[dst..dst + w * 3].copy_from_slice(&resized[src..src + w * 3]);
    }
    out
}

/// Upstream's crop-grid search: the `(columns, rows)` with `2 <= c * r <= max`
/// closest to the page's aspect. Candidates run in ascending product. On an
/// exact aspect tie, a later candidate wins when the page has the pixels for it.
pub fn uocr2_crop_grid(width: usize, height: usize, max_crops: usize) -> (usize, usize) {
    let mut candidates: Vec<(usize, usize)> = (1..=max_crops)
        .flat_map(|c| (1..=max_crops).map(move |r| (c, r)))
        .filter(|(c, r)| c * r >= 2 && c * r <= max_crops)
        .collect();
    candidates.sort_by_key(|(c, r)| c * r);
    let aspect = width as f64 / height as f64;
    let area = (width * height) as f64;
    let crop = UOCR2_CROP_SIZE as f64;
    let (mut best, mut best_diff) = ((1, 1), f64::INFINITY);
    for (c, r) in candidates {
        let diff = (aspect - c as f64 / r as f64).abs();
        if diff < best_diff {
            (best, best_diff) = ((c, r), diff);
        } else if diff == best_diff && area > 0.5 * crop * crop * (c * r) as f64 {
            best = (c, r);
        }
    }
    best
}

/// A single page, gundam mode: a page within one crop on both sides takes only
/// the global view.
pub fn uocr2_gundam_page(rgb: &[u8], width: usize, height: usize) -> Uocr2Page {
    let base = normalize(&pad_into(rgb, width, height, UOCR2_BASE_SIZE));
    if width <= UOCR2_CROP_SIZE && height <= UOCR2_CROP_SIZE {
        return Uocr2Page {
            base,
            crops: Vec::new(),
            grid: (1, 1),
        };
    }
    let (cols, rows) = uocr2_crop_grid(width, height, UOCR2_MAX_CROPS);
    let c = UOCR2_CROP_SIZE;
    let resized = resize_bicubic_pil(rgb, width, height, cols * c, rows * c);
    let mut crops = Vec::with_capacity(cols * rows);
    for r in 0..rows {
        for col in 0..cols {
            let mut crop = Vec::with_capacity(c * c * 3);
            for y in 0..c {
                let at = ((r * c + y) * cols * c + col * c) * 3;
                crop.extend_from_slice(&resized[at..at + c * 3]);
            }
            crops.push(normalize(&crop));
        }
    }
    Uocr2Page {
        base,
        crops,
        grid: (cols, rows),
    }
}

/// One page of a multi-page request: the padded global view alone.
pub fn uocr2_base_page(rgb: &[u8], width: usize, height: usize) -> Uocr2Page {
    Uocr2Page {
        base: normalize(&pad_into(rgb, width, height, UOCR2_BASE_SIZE)),
        crops: Vec::new(),
        grid: (1, 1),
    }
}

/// A request's pages, upstream's auto policy: tile a single page, and one base
/// view each when there are several. (mlx-vlm 0.7.11's `process_one` tiles every
/// page unless told otherwise. The model card and the reference `infer` do not,
/// and neither does this.)
pub fn uocr2_pages(images: &[crate::Rgb8Image]) -> Vec<Uocr2Page> {
    images
        .iter()
        .map(|i| {
            if images.len() == 1 {
                uocr2_gundam_page(&i.data, i.width, i.height)
            } else {
                uocr2_base_page(&i.data, i.width, i.height)
            }
        })
        .collect()
}

/// The token ids an image prompt is built from, off the install.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Uocr2ImageIds {
    pub bos: i32,
    pub page: i32,
    pub image: i32,
}

/// Expand a rendered prompt for these pages, upstream's `page_prompt`: no
/// `<image>` puts every page first, one expands to all of them, and N pair one
/// to one. A marker is written even where the template did not, because page2d
/// needs one per page. Any other count is refused rather than guessed at.
///
/// Unlike the rest of this crate, the error type is a plain `String`, not
/// `VisionIoError`. Kept as is so the public signature does not change.
pub fn uocr2_expand_prompt(
    rendered: &[i32],
    pages: &[Uocr2Page],
    ids: Uocr2ImageIds,
) -> Result<Vec<i32>, String> {
    let body = rendered.strip_prefix(&[ids.bos]).unwrap_or(rendered);
    let markers = body.iter().filter(|&&t| t == ids.image).count();
    if body.contains(&ids.page) {
        return Err("the prompt text already carries a <PAGE> marker".to_string());
    }
    if markers > 1 && markers != pages.len() {
        return Err(format!(
            "{markers} <image> markers for {} pages; use one marker, none, or one per page",
            pages.len()
        ));
    }
    let run = |out: &mut Vec<i32>, page: &Uocr2Page| {
        out.push(ids.page);
        out.extend(std::iter::repeat_n(ids.image, page.image_tokens()));
    };
    let mut out = Vec::with_capacity(body.len() + 2);
    out.push(ids.bos);
    if markers == 0 {
        pages.iter().for_each(|p| run(&mut out, p));
        out.extend_from_slice(body);
        return Ok(out);
    }
    let mut next = 0;
    for &token in body {
        if token != ids.image {
            out.push(token);
        } else if markers == 1 {
            pages.iter().for_each(|p| run(&mut out, p));
        } else {
            run(&mut out, &pages[next]);
            next += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDS: Uocr2ImageIds = Uocr2ImageIds {
        bos: 0,
        page: 9,
        image: 8,
    };

    fn pages(n: usize) -> Vec<Uocr2Page> {
        vec![
            Uocr2Page {
                base: Vec::new(),
                crops: Vec::new(),
                grid: (1, 1),
            };
            n
        ]
    }

    #[test]
    fn expansion_follows_page_prompt() {
        let count = |ids: &[i32], t: i32| ids.iter().filter(|&&x| x == t).count();
        // None: the pages come first, after the one BOS.
        let out = uocr2_expand_prompt(&[0, 5, 6], &pages(2), IDS).unwrap();
        assert_eq!(&out[..2], &[0, 9]);
        assert_eq!(
            (count(&out, 9), count(&out, 8), count(&out, 0)),
            (2, 546, 1)
        );
        assert_eq!(&out[out.len() - 2..], &[5, 6]);
        // One: all pages at the marker, and a template with no BOS gains it.
        let out = uocr2_expand_prompt(&[5, 8, 6], &pages(2), IDS).unwrap();
        assert_eq!(&out[..3], &[0, 5, 9]);
        assert_eq!(*out.last().unwrap(), 6);
        assert_eq!((count(&out, 9), count(&out, 8)), (2, 546));
        // N: one each, with the text between kept in place.
        let out = uocr2_expand_prompt(&[8, 5, 8], &pages(2), IDS).unwrap();
        assert_eq!(out[275], 5);
        assert_eq!(out[276], 9);
    }

    #[test]
    fn a_marker_count_with_no_reading_is_refused() {
        assert!(uocr2_expand_prompt(&[8, 8, 8], &pages(2), IDS).is_err());
        assert!(uocr2_expand_prompt(&[9, 8], &pages(1), IDS).is_err());
    }

    #[test]
    fn token_counts_are_upstreams() {
        let page = Uocr2Page {
            base: Vec::new(),
            crops: vec![Vec::new(); 12],
            grid: (3, 4),
        };
        // The reference run's own count for a 3x4 grid.
        assert_eq!(page.image_tokens(), 1513);
        let alone = Uocr2Page {
            base: Vec::new(),
            crops: Vec::new(),
            grid: (1, 1),
        };
        assert_eq!(alone.image_tokens(), 273);
    }

    #[test]
    fn the_reference_page_picks_upstreams_grid() {
        assert_eq!(uocr2_crop_grid(1000, 1400, 16), (3, 4));
    }

    #[test]
    fn a_small_page_has_no_crops() {
        let rgb = vec![200u8; 300 * 200 * 3];
        let page = uocr2_gundam_page(&rgb, 300, 200);
        assert!(page.crops.is_empty());
        assert_eq!(page.base.len(), 1024 * 1024 * 3);
        // A wide page pads top and bottom with the mean, not white or black.
        assert!((page.base[0] - (127.0 / 127.5 - 1.0)).abs() < 1e-6);
    }
}
