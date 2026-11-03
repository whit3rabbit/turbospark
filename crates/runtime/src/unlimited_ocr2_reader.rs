//! Read a document with Unlimited-OCR-2 on Metal: page layout, tower, prompt,
//! prefill, greedy decode. The front ends' one entry point
//! (`turbospark-ocr`, `ts_ocr_*`).
//!
//! One page takes gundam mode, several take a base view each, as upstream's
//! `infer` does. Decoding is greedy over upstream's opt-in 35-gram no-repeat
//! ban, which is only ever active inside a repetition loop. It looks back 128
//! tokens on one page and 1,024 on several, Baidu's own example values.

use std::path::Path;

use half::f16;
use tokenizer::MfTokenizer;
use turbospark_vision_io::{uocr2_base_page, uocr2_gundam_page, Rgb8Image, Uocr2Page};

use crate::coreml::unlimited_ocr::{banned_tokens, OcrEvent, OcrResult, OcrStats};
use crate::producer::LogitProducer;
use crate::real_forward::RealForwardRunner;

/// The prompt every released example uses, and the one the model card gives.
pub const UOCR2_PROMPT: &str = "Multi page parsing.";
const BOS: i32 = 0;
const EOS: i32 = 1;
const NGRAM: usize = 35;
const DEFAULT_MAX_NEW_ONE_PAGE: usize = 8192;
const DEFAULT_MAX_NEW_PAGES: usize = 16384;
/// Upstream's `max_length`. Every position costs 61 KiB of KV across the 12
/// layers, so this window is 2 GiB, allocated at open.
pub const UOCR2_MAX_CONTEXT: usize = 32768;

/// The whole prompt's ids for these pages: `<bos>`, then `<PAGE>` and the image
/// tokens per page, then the text.
pub fn uocr2_prompt_ids(
    pages: &[Uocr2Page],
    text_ids: &[i32],
    page_token: i32,
    image_token: i32,
) -> Vec<i32> {
    let mut ids = vec![BOS];
    for page in pages {
        ids.push(page_token);
        ids.extend(std::iter::repeat_n(image_token, page.image_tokens()));
    }
    ids.extend_from_slice(text_ids);
    ids
}

/// Layout markup out: `<BLOCK>label[x0, y0, x1, y1]<FORK>` headers go, `<PAGE>`
/// starts a new page, other `<|...|>` control tokens go. A `cross_*` or
/// `append_*` block continuing over a page break stays a separate line here;
/// upstream's converter is what stitches those.
pub fn uocr2_plain_text(raw: &str) -> String {
    let mut pages = Vec::new();
    for page in raw.split("<PAGE>") {
        let mut lines = Vec::new();
        for block in page.split("<BLOCK>") {
            let text = match block.split_once("<FORK>") {
                Some((_, text)) => text,
                None => block,
            };
            for line in text.lines() {
                let line = line.trim();
                if !line.is_empty() && !(line.starts_with("<|") && line.ends_with("|>")) {
                    lines.push(line.to_string());
                }
            }
        }
        if !lines.is_empty() {
            pages.push(lines.join("\n"));
        }
    }
    pages.join("\n\n")
}

/// An open reader. One read at a time: the KV and router state are its own.
pub struct UnlimitedOcr2Reader {
    runner: RealForwardRunner,
    vocab: usize,
    tokenizer: MfTokenizer,
    text_ids: Vec<i32>,
    page_token: i32,
    image_token: i32,
}

impl UnlimitedOcr2Reader {
    /// `arch` is the install's own, from `repack::peek_manifest_arch`.
    pub fn open(dir: &Path, arch: model_io::ArchConfig) -> Result<Self, String> {
        if arch.family != model_io::ModelFamily::UnlimitedOcr2 {
            return Err(format!(
                "{} is a {} install, not unlimited_ocr2",
                dir.display(),
                arch.family.as_str()
            ));
        }
        let vocab = arch.vocab_size as usize;
        let dash =
            model_io::DashConfig::load(dir, arch.num_layers as usize, arch.head_dim as usize)?;
        let runner = RealForwardRunner::open_with_max_context(dir, arch, UOCR2_MAX_CONTEXT)
            .map_err(|e| e.to_string())?;
        let tokenizer = MfTokenizer::load_from_dir(dir).map_err(|e| e.to_string())?;
        let text_ids = tokenizer.encode(UOCR2_PROMPT, false);
        Ok(Self {
            runner,
            vocab,
            tokenizer,
            text_ids,
            page_token: dash.page_token_id,
            image_token: dash.image_token_id,
        })
    }

    /// Reads `images` as one document. `progress` returning `false` cancels.
    pub fn read(
        &mut self,
        images: &[Rgb8Image],
        max_new_tokens: Option<usize>,
        mut progress: impl FnMut(OcrEvent) -> bool,
    ) -> Result<OcrResult, String> {
        if images.is_empty() {
            return Err("no pages".into());
        }
        let cancelled = || "cancelled".to_string();
        let started = std::time::Instant::now();
        let single = images.len() == 1;
        let mut pages = Vec::with_capacity(images.len());
        let mut rows: Vec<f16> = Vec::new();
        for (index, image) in images.iter().enumerate() {
            let page = if single {
                uocr2_gundam_page(&image.data, image.width, image.height)
            } else {
                uocr2_base_page(&image.data, image.width, image.height)
            };
            rows.extend(
                self.runner
                    .encode_unlimited_ocr2_page(&page)
                    .map_err(|e| e.to_string())?,
            );
            pages.push(page);
            if !progress(OcrEvent::PageEncoded { index }) {
                return Err(cancelled());
            }
        }
        let encode_ms = started.elapsed().as_secs_f64() * 1e3;

        let ids = uocr2_prompt_ids(&pages, &self.text_ids, self.page_token, self.image_token);
        let budget = max_new_tokens.unwrap_or(if single {
            DEFAULT_MAX_NEW_ONE_PAGE
        } else {
            DEFAULT_MAX_NEW_PAGES
        });
        if ids.len() + budget > UOCR2_MAX_CONTEXT {
            let room = UOCR2_MAX_CONTEXT.saturating_sub(ids.len());
            if room == 0 {
                return Err(format!(
                    "{} pages need {} prompt tokens, past the {UOCR2_MAX_CONTEXT}-token window",
                    images.len(),
                    ids.len()
                ));
            }
        }
        let budget = budget.min(UOCR2_MAX_CONTEXT - ids.len());

        self.runner
            .set_unlimited_ocr2_prompt(&ids, rows)
            .map_err(|e| e.to_string())?;
        self.runner.reset();
        let started = std::time::Instant::now();
        let mut logits = vec![f16::from_f32(0.0); self.vocab];
        let last = ids.len() - 1;
        for (position, &token) in ids[..last].iter().enumerate() {
            self.runner
                .produce_prefill(token, position, &mut logits)
                .map_err(|e| e.to_string())?;
            // A prefill can run many seconds, so a Stop has to reach it.
            if position % 64 == 0 && !progress(OcrEvent::Token { emitted: 0, budget }) {
                return Err(cancelled());
            }
        }
        self.runner
            .produce(ids[last], last, &mut logits)
            .map_err(|e| e.to_string())?;
        let prefill_ms = started.elapsed().as_secs_f64() * 1e3;
        if !progress(OcrEvent::Prefilled {
            tokens: ids.len(),
            capacity: UOCR2_MAX_CONTEXT,
        }) {
            return Err(cancelled());
        }

        let started = std::time::Instant::now();
        let window = if single { 128 } else { 1024 };
        let mut sequence: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
        let mut output: Vec<i32> = Vec::new();
        let mut stopped_at_eos = false;
        for emitted in 0..budget {
            let banned = banned_tokens(&sequence, NGRAM, window);
            // Lowest index on a tie, the engine sampler's rule. FP16 logits at
            // this magnitude step by 0.25, so true ties do happen on a box's
            // digits, and `max_by` alone would keep the last one.
            let next = logits
                .iter()
                .enumerate()
                .filter(|(i, _)| !banned.contains(&(*i as u32)))
                .fold(None::<(usize, f32)>, |best, (i, v)| {
                    let v = v.to_f32();
                    match best {
                        Some((_, b)) if b >= v => best,
                        _ => Some((i, v)),
                    }
                })
                .map(|(i, _)| i as i32)
                .ok_or("every token is banned")?;
            if next == EOS {
                stopped_at_eos = true;
                break;
            }
            output.push(next);
            sequence.push(next as u32);
            if !progress(OcrEvent::Token {
                emitted: emitted + 1,
                budget,
            }) {
                return Err(cancelled());
            }
            if emitted + 1 == budget {
                break;
            }
            self.runner
                .produce(next, ids.len() + emitted, &mut logits)
                .map_err(|e| e.to_string())?;
        }
        let decode_ms = started.elapsed().as_secs_f64() * 1e3;
        // The next read installs its own prompt, and a text prompt on this
        // runner must not inherit this one's pages.
        self.runner.clear_unlimited_ocr2_prompt();
        Ok(OcrResult {
            text: self.tokenizer.decode(&output, false),
            stats: OcrStats {
                pages: images.len(),
                prompt_tokens: ids.len(),
                capacity: UOCR2_MAX_CONTEXT,
                output_tokens: output.len(),
                encode_ms,
                prefill_ms,
                decode_ms,
                stopped_at_eos,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_drops_the_layout_markup() {
        let raw = "<PAGE><BLOCK>text[76, 64, 578, 90]<FORK>Quarterly Report\n\
                   <BLOCK>text[75, 152, 993, 269]<FORK>Second block.\n\
                   <PAGE><BLOCK>cross_text[1, 2, 3, 4]<FORK>Next page.";
        assert_eq!(
            uocr2_plain_text(raw),
            "Quarterly Report\nSecond block.\n\nNext page."
        );
    }

    #[test]
    fn the_prompt_marks_each_page() {
        let page = Uocr2Page {
            base: Vec::new(),
            crops: Vec::new(),
            grid: (1, 1),
        };
        let ids = uocr2_prompt_ids(&[page.clone(), page], &[7, 8], 99, 55);
        assert_eq!(ids.len(), 1 + 2 * (1 + 273) + 2);
        assert_eq!(&ids[..2], &[BOS, 99]);
        assert_eq!(ids[275], 99);
        assert_eq!(&ids[ids.len() - 2..], &[7, 8]);
    }
}
