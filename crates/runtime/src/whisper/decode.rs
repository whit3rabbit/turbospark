//! Greedy whisper decoding: the SOT grammar prompt, language selection and
//! auto-detection, special-token suppression, and the incremental decoder
//! loop over the reference kernels.
//!
//! Determinism contract: the same PCM and language produce the same tokens,
//! always. Greedy argmax with a fixed suppression mask has no sampling and
//! no tie-breaking beyond f32 argmax order, which is deterministic for a
//! fixed build.
//!
//! Position bookkeeping: decoder position `p` is fed the embedding of the
//! token at `p` plus positional row `p`, and the layer stack caches that
//! token's K/V at slot `p`. The stream state that leaves the stack after
//! position `p` produces the logits for the token at position `p + 1`.

use compute::whisper::{
    cross_kv, whisper_decoder_layer_step, WhisperCrossKv, WhisperDecoderLayerWeights, WhisperSelfKv,
};
use model_io::whisper_config::WhisperSpecialTokens;

use super::WhisperRunner;

/// The SOT grammar prefix a window's decode starts from. Multilingual:
/// `[sot, language, <|transcribe|>, <|notimestamps|>]`. English-only: the
/// reference omits the language and task block entirely -- there is no
/// language token configured, so the prompt is `[sot, <|notimestamps|>]`.
pub fn build_prompt(tokens: &WhisperSpecialTokens, language_token: u32) -> Vec<u32> {
    if tokens.multilingual {
        vec![
            tokens.sot,
            language_token,
            tokens.transcribe,
            tokens.no_timestamps,
        ]
    } else {
        vec![tokens.sot, tokens.no_timestamps]
    }
}

/// How a window's decode stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted `<|endoftext|>`.
    EndOfText,
    /// The token budget (`max_target_positions` minus the prompt) ran out.
    TokenBudget,
    /// The same token repeated eight times in a row: a stuck greedy loop,
    /// truncated rather than looped forever.
    RepetitionLimit,
}

/// Runs the decoder stack for one position: every layer consumes the
/// stream row, appends its self-attention K/V, and hands the result on.
fn step_layers(
    runner: &WhisperRunner,
    input: &[f32],
    pos: usize,
    state: &mut [WhisperSelfKv],
    cross: &[WhisperCrossKv],
) -> Result<Vec<f32>, String> {
    let d = runner.config.d_model;
    let heads = runner.config.decoder_attention_heads();
    let eps = runner.config.layer_norm_eps;
    let mut h = input.to_vec();
    for (i, layer) in runner.weights.dec_layers.iter().enumerate() {
        let w = borrow_decoder(layer);
        h = whisper_decoder_layer_step(&h, pos, &w, &mut state[i], &cross[i], d, heads, eps);
    }
    Ok(h)
}

/// Borrows the owned decoder layer weights as the kernel's view.
pub(crate) fn borrow_decoder(
    layer: &super::weights::WhisperDecoderLayerOwned,
) -> WhisperDecoderLayerWeights<'_> {
    WhisperDecoderLayerWeights {
        self_q_weight: &layer.self_q,
        self_q_bias: &layer.self_q_bias,
        self_k_weight: &layer.self_k,
        self_v_weight: &layer.self_v,
        self_v_bias: &layer.self_v_bias,
        self_out_weight: &layer.self_out,
        self_out_bias: &layer.self_out_bias,
        ln_self_weight: &layer.ln_self_weight,
        ln_self_bias: &layer.ln_self_bias,
        cross_q_weight: &layer.cross_q,
        cross_q_bias: &layer.cross_q_bias,
        cross_k_weight: &layer.cross_k,
        cross_v_weight: &layer.cross_v,
        cross_v_bias: &layer.cross_v_bias,
        cross_out_weight: &layer.cross_out,
        cross_out_bias: &layer.cross_out_bias,
        ln_cross_weight: &layer.ln_cross_weight,
        ln_cross_bias: &layer.ln_cross_bias,
        fc1_weight: &layer.fc1,
        fc1_bias: &layer.fc1_bias,
        fc2_weight: &layer.fc2,
        fc2_bias: &layer.fc2_bias,
        ln_fc_weight: &layer.ln_fc_weight,
        ln_fc_bias: &layer.ln_fc_bias,
    }
}

/// Embeds one token and adds its positional row.
fn embed_with_position(runner: &WhisperRunner, token: u32, pos: usize) -> Result<Vec<f32>, String> {
    let d = runner.config.d_model;
    let row = token as usize * d;
    if row + d > runner.weights.embed_tokens.len() {
        return Err(format!(
            "token id {token} exceeds the embedding table ({} rows)",
            runner.weights.embed_tokens.len() / d
        ));
    }
    if (pos + 1) * d > runner.weights.dec_positions.len() {
        return Err(format!(
            "decoder position {pos} exceeds max_target_positions {}",
            runner.config.max_target_positions
        ));
    }
    let scale = runner.config.embed_scale();
    let out = runner.weights.embed_tokens[row..row + d]
        .iter()
        .zip(&runner.weights.dec_positions[pos * d..pos * d + d])
        .map(|(&e, &p)| e * scale + p)
        .collect();
    Ok(out)
}

/// The per-window decoder: cross caches from the encoded window, prompt
/// warmed through the self-attention caches, then greedy steps.
pub struct WindowDecoder<'a> {
    runner: &'a WhisperRunner,
    cross: Vec<WhisperCrossKv>,
    state: Vec<WhisperSelfKv>,
    prompt_len: usize,
    /// Stream state after the most recently fed position.
    hidden: Vec<f32>,
    next_pos: usize,
    emitted: Vec<u32>,
}

impl<'a> WindowDecoder<'a> {
    /// Prepares the decoder for one encoded window: builds the cross
    /// caches, then warms the SOT prompt through the stack. The stream
    /// state after the last prompt token is where greedy decoding starts.
    pub fn new(
        runner: &'a WhisperRunner,
        encoder_out: &[f32],
        seq: usize,
        prompt: &[u32],
    ) -> Result<Self, String> {
        let d = runner.config.d_model;
        if prompt.is_empty() || prompt.len() > runner.config.max_target_positions {
            return Err("decoder prompt must be nonempty and fit max_target_positions".to_string());
        }
        if seq == 0 || seq.checked_mul(d) != Some(encoder_out.len()) {
            return Err("encoded window shape disagrees with its sequence length".to_string());
        }
        let mut cross = Vec::with_capacity(runner.weights.dec_layers.len());
        for layer in &runner.weights.dec_layers {
            let w = borrow_decoder(layer);
            cross.push(cross_kv(encoder_out, seq, &w, d));
        }
        let mut state: Vec<WhisperSelfKv> = (0..runner.weights.dec_layers.len())
            .map(|_| WhisperSelfKv::default())
            .collect();

        let mut hidden = Vec::new();
        let mut input = embed_with_position(runner, prompt[0], 0)?;
        for pos in 0..prompt.len() {
            hidden = step_layers(runner, &input, pos, &mut state, &cross)?;
            if pos + 1 < prompt.len() {
                input = embed_with_position(runner, prompt[pos + 1], pos + 1)?;
            }
        }

        Ok(Self {
            runner,
            cross,
            state,
            prompt_len: prompt.len(),
            hidden,
            next_pos: prompt.len(),
            emitted: Vec::new(),
        })
    }

    /// Greedy decode until `<|endoftext|>`, the token budget, or the
    /// repetition limit. Returns the emitted text tokens (specials and the
    /// eot excluded).
    pub fn decode(self) -> Result<DecodedWindow, String> {
        self.decode_cancellable(&|| false)
    }
    pub fn decode_cancellable(
        mut self,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<DecodedWindow, String> {
        let tokens_cfg = self.runner.tokens;
        let vocab = self.runner.config.vocab_size;
        let budget = self
            .runner
            .config
            .max_target_positions
            .saturating_sub(self.prompt_len)
            .min(440);

        let stop = loop {
            if cancelled() {
                return Err("audio job cancelled".into());
            }
            if self.emitted.len() >= budget {
                break StopReason::TokenBudget;
            }
            let logits = self.runner.logits(&self.hidden)?;
            let next = argmax_suppressed(&logits, &tokens_cfg, vocab);
            if next == tokens_cfg.eot {
                break StopReason::EndOfText;
            }
            self.emitted.push(next);
            if self.emitted.len() >= 8
                && self.emitted[self.emitted.len() - 8..]
                    .iter()
                    .all(|&t| t == next)
            {
                break StopReason::RepetitionLimit;
            }
            if self.emitted.len() >= budget {
                break StopReason::TokenBudget;
            }
            let input = embed_with_position(self.runner, next, self.next_pos)?;
            self.hidden = step_layers(
                self.runner,
                &input,
                self.next_pos,
                &mut self.state,
                &self.cross,
            )?;
            self.next_pos += 1;
        };

        Ok(DecodedWindow {
            text_tokens: std::mem::take(&mut self.emitted),
            stop,
        })
    }
}

/// One window's decoded text tokens (specials excluded, eot excluded).
pub struct DecodedWindow {
    pub text_tokens: Vec<u32>,
    pub stop: StopReason,
}

/// Greedy argmax over the suppressed logit set: everything from `eot + 1`
/// upward is a special (SOT, language, task, timestamp, nospeech) and is
/// masked; `eot` itself stays available as the stop token. `id` 0..eot are
/// the text tokens the transcript is made of.
pub fn argmax_suppressed(logits: &[f32], tokens: &WhisperSpecialTokens, vocab: usize) -> u32 {
    let mut best: Option<(f32, u32)> = None;
    for (id, &score) in logits.iter().enumerate().take(vocab) {
        let id = id as u32;
        if id > tokens.eot {
            continue;
        }
        if best.is_none_or(|(b, _)| score > b) {
            best = Some((score, id));
        }
    }
    // The mask leaves at least eot, and logits always has vocab entries.
    best.map(|(_, id)| id).unwrap_or(tokens.eot)
}

/// Language auto-detection: one decoder step over the first window with
/// only `<|startoftranscript|>` fed, argmax restricted to the language
/// token range. Deterministic by construction.
pub fn detect_language(
    runner: &WhisperRunner,
    encoder_out: &[f32],
    seq: usize,
) -> Result<u32, String> {
    let tokens_cfg = runner.tokens;
    let mut cross = Vec::with_capacity(runner.weights.dec_layers.len());
    for layer in &runner.weights.dec_layers {
        let w = borrow_decoder(layer);
        cross.push(cross_kv(encoder_out, seq, &w, runner.config.d_model));
    }
    let mut state: Vec<WhisperSelfKv> = (0..runner.weights.dec_layers.len())
        .map(|_| WhisperSelfKv::default())
        .collect();
    let input = embed_with_position(runner, tokens_cfg.sot, 0)?;
    let hidden = step_layers(runner, &input, 0, &mut state, &cross)?;
    let logits = runner.logits(&hidden)?;
    let mut best: Option<(f32, u32)> = None;
    for id in tokens_cfg.first_language..=tokens_cfg.last_language {
        if id as usize >= logits.len() {
            break;
        }
        let score = logits[id as usize];
        if best.is_none_or(|(b, _)| score > b) {
            best = Some((score, id));
        }
    }
    best.map(|(_, id)| id).ok_or_else(|| {
        "no language tokens in the vocabulary; this looks like an English-only model, pass \"en\" explicitly".to_string()
    })
}
