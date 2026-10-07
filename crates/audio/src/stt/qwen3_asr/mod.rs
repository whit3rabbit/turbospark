//! Qwen3-ASR transcription and the shared audio encoder used by Qwen3-ForcedAligner.
//!
//! Reference: `mlx_audio/stt/models/qwen3_asr/` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

use std::path::Path;
use std::time::Instant;

use serde_json::Value;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::AddedToken;
use turbospark_model_io::safetensors::SafetensorsFile;
use turbospark_tokenizer::Tokenizer;

use crate::nn::bad_config;
use crate::quant::QuantScheme;
use crate::{Result, SpeechError};

pub use config::Qwen3Config;

pub mod checkpoint;
pub mod config;
pub(super) mod decoder;
pub mod encoder;
pub mod frontend;

/// Immutable Hugging Face checkpoint reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3AsrProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The 0.6B 8-bit profile used for initial Python and frontend parity checks.
pub const QWEN3_ASR_06B_8BIT: Qwen3AsrProfile = Qwen3AsrProfile {
    name: "Qwen3-ASR 0.6B 8-bit",
    repository: "mlx-community/Qwen3-ASR-0.6B-8bit",
    revision: "89e96d92ba34aca20b3e29fb10cc284097d1219f",
};

/// Loaded Qwen3-ASR model. The decoder prefills once and then extends
/// per-layer self-attention caches during greedy generation.
pub struct Qwen3Asr {
    config: config::Qwen3Config,
    encoder: encoder::AudioEncoder,
    decoder: decoder::Decoder,
    tokenizer: Tokenizer,
}

impl Qwen3Asr {
    /// Loads one local checkpoint directory downloaded at a pinned profile
    /// revision. Multi-shard checkpoints are not supported by this first port.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_json: Value = serde_json::from_slice(
            &std::fs::read(&config_path).map_err(|error| bad_config("config.json", error))?,
        )
        .map_err(|error| bad_config("config.json", error))?;
        let config = config::Qwen3Config::from_json(&config_json)?;
        if !config.text.tie_word_embeddings {
            return Err(SpeechError::Unsupported {
                why: "Qwen3-ASR requires tied token embeddings".into(),
            });
        }
        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let scheme = QuantScheme {
            bits: config.quant_bits,
            group_size: config.quant_group_size,
        };
        let encoder = encoder::AudioEncoder::load(&weights, &config.audio)?;
        let decoder = decoder::Decoder::load(&weights, &config.text, scheme)?;
        let tokenizer = load_tokenizer(model_dir)?;

        for (token, expected) in [
            ("<|audio_pad|>", config.audio_token_id),
            ("<|audio_start|>", config.audio_start_token_id),
            ("<|audio_end|>", config.audio_end_token_id),
        ] {
            if tokenizer.token_to_id(token).map(|id| id as i32) != Some(expected) {
                return Err(bad_config(
                    "tokenizer.json",
                    format!("{token} id does not match config.json"),
                ));
            }
        }

        Ok(Self {
            config,
            encoder,
            decoder,
            tokenizer,
        })
    }

    /// Transcribes mono PCM already sampled at 16 kHz, allowing the model to
    /// detect the spoken language.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_options(samples, None, 512)
    }

    /// Transcribes one mono 16 kHz clip, optionally prompting a supported
    /// output language. `max_tokens` bounds greedy decoding.
    pub fn transcribe_with_options(
        &self,
        samples: &[f32],
        language: Option<&str>,
        max_tokens: usize,
    ) -> Result<String> {
        if max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "Qwen3-ASR max_tokens must be positive".into(),
            });
        }
        let language = language
            .map(|requested| {
                self.config
                    .supported_languages
                    .iter()
                    .find(|known| known.eq_ignore_ascii_case(requested))
                    .map(String::as_str)
                    .ok_or_else(|| SpeechError::Input {
                        why: format!("unsupported Qwen3-ASR language: {requested}"),
                    })
            })
            .transpose()?;

        // TURBOSPARK_QWEN3_ASR_PROFILE=1 prints per-phase wall times; the
        // clock reads are negligible next to the phases they bracket.
        let profile = std::env::var("TURBOSPARK_QWEN3_ASR_PROFILE").as_deref() == Ok("1");
        let started = Instant::now();
        let features = frontend::compute_features(samples)?;
        let features_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        let audio_embeddings = self.encoder.forward(&features)?;
        let encode_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        if audio_embeddings.len() % self.decoder.hidden_size != 0 {
            return Err(SpeechError::Tensor {
                name: "audio_tower.output".into(),
                why: "encoder output is not a whole number of decoder embeddings".into(),
            });
        }
        let audio_rows = audio_embeddings.len() / self.decoder.hidden_size;
        if audio_rows == 0 {
            return Err(SpeechError::Input {
                why: "Qwen3 audio encoder produced no features".into(),
            });
        }

        let (token_ids, audio_positions) =
            prompt_token_ids(&self.config, &self.tokenizer, audio_rows, language)?;
        let mut embeddings = self.decoder.embed(&token_ids)?;
        let prompt_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        for (audio_row, &position) in audio_positions.iter().enumerate() {
            let target = position * self.decoder.hidden_size;
            let source = audio_row * self.decoder.hidden_size;
            embeddings[target..target + self.decoder.hidden_size]
                .copy_from_slice(&audio_embeddings[source..source + self.decoder.hidden_size]);
        }

        let stop_ids = decoder::chat_stop_ids(&self.tokenizer);
        let rows = embeddings.len() / self.decoder.hidden_size;
        let (last_hidden, cache) = self.decoder.prefill(&embeddings, rows);
        let prefill_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        let logits = self.decoder.logits(&last_hidden);
        let generated = decoder::greedy_generate(
            &self.decoder,
            &logits,
            cache,
            max_tokens,
            |next| decoder::is_chat_stop(&stop_ids, next),
            "Qwen3",
            |_, _| {},
        )?;
        let decode_ms = started.elapsed().as_secs_f64() * 1_000.0;
        if profile {
            eprintln!(
                "qwen3_asr_cpu features_ms={features_ms:.2} encode_ms={encode_ms:.2} \
                 prompt_ms={prompt_ms:.2} prefill_ms={prefill_ms:.2} decode_ms={decode_ms:.2} \
                 audio_rows={audio_rows} tokens={}",
                generated.len()
            );
        }

        let decoded =
            self.tokenizer
                .decode(&generated, false)
                .map_err(|error| SpeechError::Input {
                    why: format!("Qwen3 output detokenization failed: {error}"),
                })?;
        let text = if language.is_none() {
            decoded
                .find("<asr_text>")
                .map(|separator| decoded[separator + "<asr_text>".len()..].to_owned())
                .unwrap_or(decoded)
        } else {
            decoded
        };
        Ok(text.trim().to_owned())
    }
}

/// Builds the Qwen3-ASR prompt for one clip and returns its token ids with
/// the positions of the audio placeholder tokens. `audio_rows` must match
/// the encoder's frame count; the placeholder count is validated here so
/// the CPU and Metal decode paths substitute audio embeddings against the
/// same positions.
pub fn prompt_token_ids(
    config: &Qwen3Config,
    tokenizer: &Tokenizer,
    audio_rows: usize,
    language: Option<&str>,
) -> Result<(Vec<i32>, Vec<usize>)> {
    if audio_rows == 0 {
        return Err(SpeechError::Input {
            why: "Qwen3 audio encoder produced no features".into(),
        });
    }
    let assistant_prefix = language
        .map(|language| format!("language {language}<asr_text>"))
        .unwrap_or_default();
    let prompt = format!(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>{}<|audio_end|><|im_end|>\n<|im_start|>assistant\n{assistant_prefix}",
        "<|audio_pad|>".repeat(audio_rows)
    );
    let encoded = tokenizer
        .encode(prompt, false)
        .map_err(|error| SpeechError::Input {
            why: format!("Qwen3 prompt tokenization failed: {error}"),
        })?;
    let token_ids = encoded
        .get_ids()
        .iter()
        .map(|&id| {
            i32::try_from(id).map_err(|_| SpeechError::Input {
                why: "Qwen3 prompt token id exceeds signed 32-bit range".into(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let audio_positions = token_ids
        .iter()
        .enumerate()
        .filter_map(|(position, &id)| (id == config.audio_token_id).then_some(position))
        .collect::<Vec<_>>();
    if audio_positions.len() != audio_rows {
        return Err(SpeechError::Input {
            why: format!(
                "Qwen3 prompt has {} audio placeholders for {audio_rows} encoded frames",
                audio_positions.len()
            ),
        });
    }
    Ok((token_ids, audio_positions))
}

/// Detokenizes generated ids and, when no language was requested, strips
/// the scaffolding the model may echo before `<asr_text>`.
pub fn transcript_from_tokens(
    tokenizer: &Tokenizer,
    generated: &[u32],
    language: Option<&str>,
) -> Result<String> {
    let decoded = tokenizer
        .decode(generated, false)
        .map_err(|error| SpeechError::Input {
            why: format!("Qwen3 output detokenization failed: {error}"),
        })?;
    let text = if language.is_none() {
        decoded
            .find("<asr_text>")
            .map(|separator| decoded[separator + "<asr_text>".len()..].to_owned())
            .unwrap_or(decoded)
    } else {
        decoded
    };
    Ok(text.trim().to_owned())
}

pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if tokenizer_path.is_file() {
        return Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| bad_config("tokenizer.json", error));
    }

    // The pinned mlx-community repository publishes Qwen2's original BPE
    // assets instead of tokenizer.json. Rebuild its byte-level tokenizer and
    // preserve each added token's explicit ID from tokenizer_config.json.
    let config_path = model_dir.join("tokenizer_config.json");
    let config: Value = serde_json::from_slice(
        &std::fs::read(&config_path).map_err(|error| bad_config("tokenizer_config.json", error))?,
    )
    .map_err(|error| bad_config("tokenizer_config.json", error))?;
    if config.get("tokenizer_class").and_then(Value::as_str) != Some("Qwen2Tokenizer")
        || config.get("add_prefix_space").and_then(Value::as_bool) != Some(false)
    {
        return Err(SpeechError::Unsupported {
            why: "Qwen3 tokenizer fallback supports Qwen2Tokenizer without prefix space".into(),
        });
    }
    let vocab_path = model_dir.join("vocab.json");
    let merges_path = model_dir.join("merges.txt");
    let vocab = vocab_path.to_string_lossy();
    let merges = merges_path.to_string_lossy();
    let bpe = BPE::from_file(vocab.as_ref(), merges.as_ref())
        .build()
        .map_err(|error| bad_config("vocab.json/merges.txt", error))?;
    let byte_level = ByteLevel::new(false, true, true);
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_decoder(Some(byte_level));

    let entries = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .ok_or_else(|| bad_config("tokenizer_config.json", "added_tokens_decoder is missing"))?;
    let mut entries = entries
        .iter()
        .map(|(raw_id, entry)| {
            let id = raw_id
                .parse::<u32>()
                .map_err(|error| bad_config("tokenizer_config.json.added_tokens_decoder", error))?;
            let content = entry
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    bad_config(
                        "tokenizer_config.json.added_tokens_decoder",
                        format!("token {id} has no content"),
                    )
                })?
                .to_owned();
            let field = |name| entry.get(name).and_then(Value::as_bool).unwrap_or(false);
            let token = AddedToken::from(content.clone(), field("special"))
                .single_word(field("single_word"))
                .lstrip(field("lstrip"))
                .rstrip(field("rstrip"))
                .normalized(field("normalized"));
            Ok((id, content, token))
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort_by_key(|(id, _, _)| *id);

    let base_vocab = u32::try_from(tokenizer.get_vocab_size(false)).map_err(|error| {
        bad_config(
            "vocab.json",
            format!("base vocabulary is too large: {error}"),
        )
    })?;
    for (offset, (id, content, token)) in entries.into_iter().enumerate() {
        let expected_id = base_vocab + offset as u32;
        if id != expected_id {
            return Err(bad_config(
                "tokenizer_config.json.added_tokens_decoder",
                format!("expected added token id {expected_id}, found {id}"),
            ));
        }
        if token.special {
            tokenizer.add_special_tokens(&[token]);
        } else {
            tokenizer.add_tokens(&[token]);
        }
        if tokenizer.token_to_id(&content) != Some(id) {
            return Err(bad_config(
                "tokenizer_config.json.added_tokens_decoder",
                format!("added token {content} did not retain id {id}"),
            ));
        }
    }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::Qwen3Asr;

    #[test]
    #[ignore = "requires the pinned Qwen3-ASR checkpoint and spoken reference WAV"]
    fn pinned_checkpoint_transcribes_reference_wav() {
        let model_dir = std::env::var_os("TURBOSPARK_QWEN3_ASR_DIR")
            .expect("set TURBOSPARK_QWEN3_ASR_DIR to the pinned checkpoint directory");
        let audio_path = std::env::var_os("TURBOSPARK_QWEN3_ASR_WAV")
            .expect("set TURBOSPARK_QWEN3_ASR_WAV to the reference speech clip");
        let model = Qwen3Asr::load(std::path::Path::new(&model_dir)).unwrap();
        let audio = turbospark_audio::wav::read_wav_f32(std::path::Path::new(&audio_path)).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        let text = model
            .transcribe_with_options(&audio.samples, None, 32)
            .unwrap();
        assert_eq!(text, "The quick brown fox jumps over the lazy dog.");
    }

    #[test]
    #[ignore = "requires the pinned Qwen3-ASR checkpoint and spoken reference WAV"]
    fn pinned_checkpoint_runs_one_greedy_decode_step() {
        let model_dir = std::env::var_os("TURBOSPARK_QWEN3_ASR_DIR")
            .expect("set TURBOSPARK_QWEN3_ASR_DIR to the pinned checkpoint directory");
        let audio_path = std::env::var_os("TURBOSPARK_QWEN3_ASR_WAV")
            .expect("set TURBOSPARK_QWEN3_ASR_WAV to the reference speech clip");
        let model = Qwen3Asr::load(std::path::Path::new(&model_dir)).unwrap();
        let audio = turbospark_audio::wav::read_wav_f32(std::path::Path::new(&audio_path)).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        let text = model
            .transcribe_with_options(&audio.samples[..16_000], None, 1)
            .unwrap();
        eprintln!("one-token Qwen3-ASR output: {text:?}");
        assert!(!text.is_empty(), "checkpoint stopped before emitting text");
    }
}
