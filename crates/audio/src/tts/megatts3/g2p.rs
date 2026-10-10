//! MegaTTS3's grapheme-to-phoneme model (`g2p()` in `frontend_function.py`).
//!
//! A Qwen2-0.5B causal LM fine-tuned to write phonemes: the prompt is
//! `<BOT>` + text + `<BOS>` followed by one speech token (`145 + speech_start`,
//! the `sil` phone), and the model emits phone tokens (`speech_start + id`)
//! until `800 + 1 + speech_start`. `speech_start` is the id of the added token
//! `<Reserved_TTS_0>`. Decoding is greedy (`do_sample=True, top_k=1`, and the
//! checkpoint's generation config adds no processors), capped at 256 new tokens.
//!
//! Like the reference, the returned phone sequence starts with the prompt's
//! `sil` token and always drops the final generated token (the EOS, or a real
//! token if the cap was hit). Weights are the HF checkpoint as shipped
//! (bf16 on disk, widened to f32); it runs on the shared dense backbone in
//! [`crate::lm`].

use std::path::Path;

use tokenizers::Tokenizer;

use crate::lm::{make_prompt_cache, qwen2, Backbone, Input, Weights};
use crate::{Result, SpeechError};

/// Phones the aligner and G2P share: ids below 200 are plain phones.
const MAX_PLAIN_PHONE: i32 = 173;
/// `sil` is phone 145; the prompt ends with its speech token.
const SIL_PHONE: i32 = 145;
/// The EOS speech token is `800 + 1` past the speech base.
const EOS_OFFSET: i32 = 801;
pub const MAX_NEW_TOKENS: usize = 256;
/// Longest prompt accepted; the chunker keeps chunks far below this.
const MAX_PROMPT_TOKENS: usize = 2048;
const BOT: &str = "<BOT>";
const BOS: &str = "<BOS>";
const SPEECH_START_TOKEN: &str = "<Reserved_TTS_0>";

pub struct G2p {
    tokenizer: Tokenizer,
    backbone: Backbone,
    speech_start: i32,
}

fn tok_err(e: impl std::fmt::Display) -> SpeechError {
    SpeechError::Audio(format!("g2p tokenizer: {e}"))
}

impl G2p {
    /// Opens `dir` holding `tokenizer.json`, `config.json`, and
    /// `model.safetensors` (the converted directory's `g2p/`).
    pub fn open(dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(tok_err)?;
        let weights = Weights::open(dir)?;
        Self::load(tokenizer, &weights)
    }

    pub fn load(tokenizer: Tokenizer, weights: &Weights) -> Result<Self> {
        let backbone = qwen2::load(weights, weights.config())?;
        let first = tokenizer
            .encode(SPEECH_START_TOKEN, true)
            .map_err(tok_err)?;
        let speech_start = match first.get_ids() {
            [id, ..] => *id as i32,
            [] => {
                return Err(SpeechError::Tensor {
                    name: SPEECH_START_TOKEN.to_string(),
                    why: "tokenizer produced no ids".to_string(),
                })
            }
        };
        let need = speech_start as usize + EOS_OFFSET as usize + 1;
        if backbone.config.vocab_size < need {
            return Err(SpeechError::BadConfig {
                field: "vocab_size".to_string(),
                why: format!(
                    "{} is too small for speech tokens up to {need}",
                    backbone.config.vocab_size
                ),
            });
        }
        Ok(Self {
            tokenizer,
            backbone,
            speech_start,
        })
    }

    pub fn speech_start(&self) -> i32 {
        self.speech_start
    }

    /// Token ids of `<BOT>` + text + `<BOS>` (the reference's `txt_token`).
    pub fn text_ids(&self, text: &str) -> Result<Vec<i32>> {
        let ids = self
            .tokenizer
            .encode(format!("{BOT}{text}{BOS}"), true)
            .map_err(tok_err)?;
        Ok(ids.get_ids().iter().map(|&i| i as i32).collect())
    }

    /// Greedy generation (HF `generate`, `top_k=1`): returns the prompt followed
    /// by the generated tokens, the EOS included when it was produced.
    pub fn generate(&mut self, prompt: &[i32], max_new: usize, eos: i32) -> Result<Vec<i32>> {
        if prompt.is_empty() || prompt.len() > MAX_PROMPT_TOKENS {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt of {} tokens is outside 1..={MAX_PROMPT_TOKENS}",
                    prompt.len()
                ),
            });
        }
        let vocab = self.backbone.config.vocab_size;
        if prompt.iter().any(|&t| t < 0 || t as usize >= vocab) {
            return Err(SpeechError::Input {
                why: "prompt token outside the vocabulary".to_string(),
            });
        }
        let mut caches = make_prompt_cache(self.backbone.config.num_hidden_layers);
        let mut out = prompt.to_vec();
        let mut logits = self
            .backbone
            .forward(Input::Tokens(prompt), Some(&mut caches))?;
        for _ in 0..max_new {
            let last = &logits[logits.len() - vocab..];
            // torch.argmax: the first maximum wins.
            let mut best = 0usize;
            for (i, &v) in last.iter().enumerate() {
                if v > last[best] {
                    best = i;
                }
            }
            let tok = best as i32;
            out.push(tok);
            if tok == eos {
                break;
            }
            logits = self
                .backbone
                .forward(Input::Tokens(&[tok]), Some(&mut caches))?;
        }
        Ok(out)
    }

    /// Phones and tones for one text chunk.
    pub fn phonemize(&mut self, text: &str) -> Result<(Vec<i32>, Vec<i32>)> {
        let txt = self.text_ids(text)?;
        let input = generation_input(&txt, self.speech_start);
        let out = self.generate(&input, MAX_NEW_TOKENS, EOS_OFFSET + self.speech_start)?;
        phones_from_output(&out, txt.len(), self.speech_start)
    }
}

/// The generation prompt: the text tokens followed by the `sil` speech token.
pub(super) fn generation_input(txt: &[i32], speech_start: i32) -> Vec<i32> {
    let mut input = txt.to_vec();
    input.push(SIL_PHONE + speech_start);
    input
}

/// `outputs[:, len(txt_token):-1] - speech_start`, then `split_ph`: keeps the
/// prompt's `sil` token, drops the final token (the EOS, or a real token when
/// the budget ran out), and maps the rest to phones and tones.
pub(super) fn phones_from_output(
    out: &[i32],
    txt_len: usize,
    speech_start: i32,
) -> Result<(Vec<i32>, Vec<i32>)> {
    if out.len() <= txt_len + 1 {
        return Err(SpeechError::Input {
            why: "g2p output is shorter than its prompt".to_string(),
        });
    }
    let ids: Vec<i32> = out[txt_len..out.len() - 1]
        .iter()
        .map(|&t| t - speech_start)
        .collect();
    phones_from_ids(&ids)
}

/// `split_ph`: Chinese phones fold phone and tone into ids `200..=788`
/// (`(id - 201) / 6 + 3`, tone from the remainder); every other id keeps tone 3.
/// Ids no embedding table could hold are errors rather than silent garbage.
pub fn phones_from_ids(ids: &[i32]) -> Result<(Vec<i32>, Vec<i32>)> {
    let mut ph = Vec::with_capacity(ids.len());
    let mut tone = Vec::with_capacity(ids.len());
    for &id in ids {
        if (200..=788).contains(&id) {
            ph.push((id - 200 - 1).div_euclid(6) + 3);
            let t = (id - 200 - 1).rem_euclid(6) + 1;
            tone.push(if t == 1 { 4 } else { t + 9 });
        } else if (1..=MAX_PLAIN_PHONE).contains(&id) {
            ph.push(id);
            tone.push(3);
        } else {
            return Err(SpeechError::Input {
                why: format!("g2p produced token {id}, which is not a phone id"),
            });
        }
    }
    if ph.is_empty() {
        return Err(SpeechError::Input {
            why: "g2p produced no phones".to_string(),
        });
    }
    Ok((ph, tone))
}
