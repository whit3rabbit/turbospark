//! Self-contained US English frontend and checked Kokoro sentence contract.
//!
//! Dictionary, stress, inflection and POS machinery is adapted from the pinned
//! Misaki Rust frontend. Compound and punctuation-context behavior follows
//! hexgrad/misaki fba1236595f2d2bf21d414ba6e57d25256afada3 (Apache-2.0,
//! see licenses/MISAKI-APACHE-2.0.txt). No process or Python runtime is used.
mod data;
mod lexicon;
mod normalize;
mod tagger;

use crate::{Result, SpeechError};
use lexicon::{Lexicon, TokenContext};
use std::{collections::HashMap, path::Path, sync::OnceLock};
use tagger::PerceptronTagger;
use turbospark_model_io::safetensors::SafetensorsFile;

/// Two blank tokens also occupy the checkpoint's 512 position table.
pub const MAX_PHONEME_CHARS: usize = 510;
const STYLE_WIDTH: usize = 256;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Voice {
    #[default]
    AfHeart,
}
impl Voice {
    pub fn as_str(self) -> &'static str {
        "af_heart"
    }
}
impl std::str::FromStr for Voice {
    type Err = SpeechError;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "af_heart" => Ok(Self::AfHeart),
            _ => Err(SpeechError::Unsupported {
                why: format!("voice {value:?}; supported voice is af_heart"),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub language: String,
    pub voice: Voice,
    pub speed: f32,
}
impl SynthesisRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: "en-US".into(),
            voice: Voice::AfHeart,
            speed: 1.0,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.language != "en-US" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "language {:?}; portable Kokoro frontend supports en-US",
                    self.language
                ),
            });
        }
        if !self.speed.is_finite() || self.speed <= 0.0 {
            return Err(input("speed must be finite and positive"));
        }
        normalize::validate_text(&self.text)
    }
}

/// A vocabulary checked against both model embedding tables and positions.
#[derive(Debug, Clone)]
pub struct PhonemeVocabulary {
    map: HashMap<char, u32>,
    max_chars: usize,
}
impl PhonemeVocabulary {
    pub fn new(
        map: HashMap<char, u32>,
        bert_rows: usize,
        text_rows: usize,
        max_positions: usize,
    ) -> Result<Self> {
        if map.is_empty() || bert_rows == 0 || text_rows == 0 || max_positions < 3 {
            return Err(input("Kokoro vocabulary and embedding tables must be nonempty, with at least three positions"));
        }
        for (&phone, &id) in &map {
            if id == 0 || id as usize >= bert_rows || id as usize >= text_rows {
                return Err(input(&format!("phone U+{:04X} has ID {id}, outside nonblank embedding rows (BERT {bert_rows}, text {text_rows})",phone as u32)));
            }
        }
        Ok(Self {
            map,
            max_chars: MAX_PHONEME_CHARS.min(max_positions - 2),
        })
    }
    pub fn from_config(
        config: &serde_json::Value,
        bert_rows: usize,
        text_rows: usize,
        max_positions: usize,
    ) -> Result<Self> {
        let entries = config
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| input("config.vocab must be a phone-to-ID object"))?;
        let mut map = HashMap::new();
        for (phone, id) in entries {
            let mut chars = phone.chars();
            let ch = chars
                .next()
                .filter(|_| chars.next().is_none())
                .ok_or_else(|| {
                    input(&format!(
                        "vocab key {phone:?} must be one Unicode character"
                    ))
                })?;
            let id = id
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| input(&format!("vocab ID for {phone:?} must fit u32")))?;
            map.insert(ch, id);
        }
        Self::new(map, bert_rows, text_rows, max_positions)
    }
    fn checked_ids(&self, phonemes: &str) -> Result<Vec<u32>> {
        let count = phonemes.chars().count();
        if count == 0 || count > self.max_chars || phonemes.trim().is_empty() {
            return Err(input(&format!("segment has {count} phoneme characters; expected 1 through {} before two blank tokens",self.max_chars)));
        }
        let mut ids = Vec::with_capacity(count + 2);
        ids.push(0);
        for phone in phonemes.chars() {
            ids.push(*self.map.get(&phone).ok_or_else(||SpeechError::Unsupported {why:format!("phoneme U+{:04X} is absent from this Kokoro vocabulary; use supported US English text",phone as u32)})?);
        }
        ids.push(0);
        Ok(ids)
    }
}

/// Fields are private so a runtime receives only complete checked segments.
#[derive(Debug, Clone)]
pub struct SynthesisSegment {
    text: String,
    phonemes: String,
    ids: Vec<u32>,
}
impl SynthesisSegment {
    pub fn from_phonemes(phonemes: &str, vocab: &PhonemeVocabulary) -> Result<Self> {
        Ok(Self {
            text: String::new(),
            phonemes: phonemes.into(),
            ids: vocab.checked_ids(phonemes)?,
        })
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn phonemes(&self) -> &str {
        &self.phonemes
    }
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    /// The reference indexes the voice by phoneme character count minus one.
    pub fn style_row(&self) -> usize {
        self.phonemes.chars().count() - 1
    }
}

pub struct VoicePack {
    voice: Voice,
    values: Vec<f32>,
}
impl VoicePack {
    pub fn from_values(voice: Voice, shape: &[usize], values: Vec<f32>) -> Result<Self> {
        if shape != [MAX_PHONEME_CHARS, 1, STYLE_WIDTH]
            || values.len() != MAX_PHONEME_CHARS * STYLE_WIDTH
        {
            return Err(input(&format!("{} voice pack must have shape [510, 1, 256] and 130560 values; got {shape:?} and {} values",voice.as_str(),values.len())));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(input("voice pack contains nonfinite values"));
        }
        Ok(Self { voice, values })
    }
    pub fn open(path: &Path, voice: Voice) -> Result<Self> {
        let file = SafetensorsFile::open(path)?;
        let name = voice.as_str();
        let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
            name: name.into(),
            why: "voice tensor missing".into(),
        })?;
        Self::from_values(voice, &descriptor.shape, file.load_as_f32(name)?)
    }
    pub fn voice(&self) -> Voice {
        self.voice
    }
    pub fn style_for(&self, segment: &SynthesisSegment) -> &[f32] {
        let start = segment.style_row() * STYLE_WIDTH;
        &self.values[start..start + STYLE_WIDTH]
    }
}

struct Engine {
    lexicon: Lexicon,
    tagger: PerceptronTagger,
}
static ENGINE: OnceLock<Engine> = OnceLock::new();
/// Bundled English resources are parsed once and shared across requests.
#[derive(Default)]
pub struct EnglishFrontend;
impl EnglishFrontend {
    pub fn new() -> Self {
        Self
    }
    pub fn prepare(
        &self,
        request: &SynthesisRequest,
        vocab: &PhonemeVocabulary,
    ) -> Result<Vec<SynthesisSegment>> {
        request.validate()?;
        let engine = ENGINE.get_or_init(|| Engine {
            lexicon: Lexicon::new(),
            tagger: PerceptronTagger::new(
                include_str!("resources/pos_weights.json"),
                include_str!("resources/pos_classes.txt"),
                include_str!("resources/pos_tags.json"),
            ),
        });
        let normalized = normalize::text(&request.text);
        let mut tokens = normalize::tokens(&normalized)?;
        let words: Vec<_> = tokens.iter().map(|t| t.word.as_str()).collect();
        let tags = engine.tagger.tag(&words);
        let mut opening_quote = true;
        for (token, tag) in tokens.iter_mut().zip(tags) {
            token.tag = tag.tag;
            if token.word == "\"" {
                // The trained vocabulary distinguishes opening and closing quotes.
                token.phones = if opening_quote {
                    "\u{201c}"
                } else {
                    "\u{201d}"
                }
                .into();
                opening_quote = !opening_quote;
            }
        }
        correct_context_tags(&mut tokens, &engine.lexicon);
        let mut future = TokenContext::default();
        for token in tokens.iter_mut().rev() {
            let phones = if let Some(number) = &token.spoken {
                pronounce_number(number, &engine.lexicon)?
            } else if token.punctuation {
                if token.phones.is_empty() {
                    token.word.clone()
                } else {
                    token.phones.clone()
                }
            } else {
                let stress = word_stress(&token.word, &engine.lexicon);
                engine.lexicon
                    .get_word(&token.word, &token.tag, stress, Some(&future))
                    .map(|p| p.0)
                    .or_else(|| {
                        if token.word.contains('-') {
                            pronounce_compound(&token.word, &token.tag, &engine.lexicon, &future)
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        if token.word.chars().all(|c| c.is_ascii_alphabetic()) {
                            engine.lexicon.get_nnp(&token.word).map(|p| p.0)
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| SpeechError::Unsupported {
                        why: format!(
                            "cannot phonemize {:?}; only ASCII English words, numbers and supported punctuation can be pronounced",
                            token.word
                        ),
                    })?
            };
            // Kokoro v1 uses T for the American flap and t for the glottal stop.
            let phones = phones.replace('\u{27e}', "T").replace('\u{294}', "t");
            if phones.chars().count() > vocab.max_chars {
                return Err(input(&format!(
                    "word {:?} exceeds {} phoneme characters; shorten or separate it",
                    token.word, vocab.max_chars
                )));
            }
            // Sound, rather than spelling, controls articles before hour/university.
            future.future_vowel = next_vowel(&phones, future.future_vowel);
            future.future_to = token.word.eq_ignore_ascii_case("to");
            token.phones = phones;
        }
        chunk(&normalized, &tokens, vocab)
    }
}

fn pronounce_number(spoken: &str, lexicon: &Lexicon) -> Result<String> {
    spoken
        .split_whitespace()
        .map(|w| {
            lexicon
                // Misaki deliberately removes stress from decimal "point".
                .get_word(w, "NN", if w == "point" { Some(-2.0) } else { None }, None)
                .map(|p| p.0)
                .ok_or_else(|| input(&format!("cannot pronounce normalized number word {w:?}")))
        })
        .collect::<Result<Vec<_>>>()
        .map(|words| words.join(" "))
}

fn next_vowel(phones: &str, previous: Option<bool>) -> Option<bool> {
    for ch in phones.chars() {
        if ";:,.!?\u{2014}\u{2026}".contains(ch) {
            return None;
        }
        if "AIOQWYaiu\u{e6}\u{251}\u{252}\u{254}\u{259}\u{25b}\u{25c}\u{26a}\u{28a}\u{28c}\u{1d7b}"
            .contains(ch)
        {
            return Some(true);
        }
        if "bdfhjklmnpstvwz\u{f0}\u{14b}\u{261}\u{279}\u{27e}\u{283}\u{292}\u{2a4}\u{2a7}\u{3b8}"
            .contains(ch)
        {
            return Some(false);
        }
    }
    // Quotes and parentheses do not interrupt the following sound's context.
    previous
}

fn word_stress(word: &str, lexicon: &Lexicon) -> Option<f64> {
    if word == word.to_lowercase() {
        None
    } else if word == word.to_uppercase() {
        Some(lexicon.cap_stresses.1)
    } else {
        Some(lexicon.cap_stresses.0)
    }
}

fn pronounce_compound(
    word: &str,
    tag: &str,
    lexicon: &Lexicon,
    future: &TokenContext,
) -> Option<String> {
    let parts: Vec<_> = word.split('-').collect();
    if parts.len() > MAX_PHONEME_CHARS
        || parts
            .iter()
            .any(|p| p.is_empty() || !p.chars().all(|c| c.is_ascii_alphabetic()))
    {
        return None;
    }
    let mut phones = Vec::new();
    let mut context = future.clone();
    let mut right = parts.len();
    while right > 0 {
        // Resolve the longest dictionary span from the right, as Misaki does.
        // This preserves lexical compound stress, such as the entry "one-two".
        let mut matched = None;
        for left in 0..right {
            let span = parts[left..right].join("-");
            if let Some((ps, _)) =
                lexicon.get_word(&span, tag, word_stress(&span, lexicon), Some(&context))
            {
                matched = Some((left, span, ps));
                break;
            }
        }
        let (left, span, ps) = matched.or_else(|| {
            let part = parts[right - 1];
            lexicon
                .get_nnp(part)
                .map(|(ps, _)| (right - 1, part.into(), ps))
        })?;
        context.future_vowel = next_vowel(&ps, context.future_vowel);
        context.future_to = span.eq_ignore_ascii_case("to");
        phones.push((span, ps));
        right = left;
    }
    phones.reverse();
    // Misaki's compound resolver demotes the weaker half when most parts
    // carry primary stress. Weight ties retain their left-to-right order.
    if phones.len() == 2 && phones[0].0.len() == 1 {
        phones[1].1 = lexicon.apply_stress(&phones[1].1, Some(-0.5));
    } else if phones.iter().filter(|(_, p)| p.contains('\u{2c8}')).count()
        > phones.len().div_ceil(2)
    {
        let mut weights: Vec<_> = phones
            .iter()
            .enumerate()
            .map(|(i, (_, p))| {
                let weight: usize = p
                    .chars()
                    .map(|c| {
                        if "AIOQWY\u{2a4}\u{2a7}".contains(c) {
                            2
                        } else {
                            1
                        }
                    })
                    .sum();
                ((p.contains('\u{2c8}'), weight, i), i)
            })
            .collect();
        weights.sort_unstable();
        for (_, i) in weights.into_iter().take(phones.len() / 2) {
            phones[i].1 = lexicon.apply_stress(&phones[i].1, Some(-0.5));
        }
    }
    Some(phones.into_iter().map(|(_, p)| p).collect())
}

fn correct_context_tags(tokens: &mut [normalize::Token], lexicon: &Lexicon) {
    for i in 0..tokens.len() {
        if tokens[i].punctuation || tokens[i].spoken.is_some() {
            continue;
        }
        let word = tokens[i].word.to_lowercase();
        let previous = if i == 0 {
            ""
        } else {
            tokens[i - 1].word.as_str()
        };
        let next = tokens
            .get(i + 1)
            .map(|t| t.word.to_lowercase())
            .unwrap_or_default();
        let boundary = i == 0 || matches!(previous, "." | "!" | "?");
        let heteronym =
            lexicon.lookup(&word, "VB", None, None) != lexicon.lookup(&word, "NN", None, None);
        if previous.eq_ignore_ascii_case("please")
            || (boundary
                && heteronym
                && matches!(next.as_str(), "the" | "a" | "an" | "it" | "this" | "that"))
        {
            tokens[i].tag = "VB".into();
        }
        // Canonical dictionaries have a past-tense VBP entry for read. Retain
        // past cues, but choose the base form for explicit habitual/present cues.
        if word == "read"
            && (matches!(next.as_str(), "every" | "daily" | "now" | "today")
                || matches!(previous, "will" | "can" | "to"))
        {
            tokens[i].tag = "VB".into();
        }
        if word == "bass"
            && matches!(
                next.as_str(),
                "guitar" | "guitars" | "drum" | "drums" | "clef" | "note" | "notes"
            )
        {
            tokens[i].tag = "JJ".into();
        }
    }
}

fn chunk(
    text: &str,
    tokens: &[normalize::Token],
    vocab: &PhonemeVocabulary,
) -> Result<Vec<SynthesisSegment>> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut phones = String::new();
    let mut count = 0;
    let mut sentence_has_end = false;
    for (i, token) in tokens.iter().enumerate() {
        let gap = if i > start && tokens[i - 1].end < token.start {
            " "
        } else {
            ""
        };
        let added = gap.len() + token.phones.chars().count();
        if count + added > vocab.max_chars && i > start {
            emit_segment(text, tokens, start, i, &phones, vocab, &mut segments)?;
            start = i;
            phones.clear();
            count = 0;
            sentence_has_end = false;
        } else {
            phones.push_str(gap);
            count += gap.len();
        }
        phones.push_str(&token.phones);
        count += token.phones.chars().count();
        sentence_has_end |= matches!(token.word.as_str(), "." | "!" | "?");
        // Keep an adjacent terminal run and all closing delimiters attached
        // so synthesis never receives a standalone punctuation fragment.
        let continuing = tokens.get(i + 1).is_some_and(|next| {
            token.end == next.start && matches!(next.word.as_str(), "." | "!" | "?" | ")" | "\"")
        });
        if sentence_has_end && !continuing {
            emit_segment(text, tokens, start, i + 1, &phones, vocab, &mut segments)?;
            start = i + 1;
            phones.clear();
            count = 0;
            sentence_has_end = false;
        }
    }
    if start < tokens.len() {
        emit_segment(
            text,
            tokens,
            start,
            tokens.len(),
            &phones,
            vocab,
            &mut segments,
        )?;
    }
    if segments.is_empty() {
        return Err(input("text contains no pronounceable segment"));
    }
    Ok(segments)
}
fn emit_segment(
    text: &str,
    tokens: &[normalize::Token],
    start: usize,
    end: usize,
    phones: &str,
    vocab: &PhonemeVocabulary,
    out: &mut Vec<SynthesisSegment>,
) -> Result<()> {
    let mut segment = SynthesisSegment::from_phonemes(phones, vocab)?;
    segment.text = text[tokens[start].start..tokens[end - 1].end].into();
    out.push(segment);
    Ok(())
}
fn input(why: &str) -> SpeechError {
    SpeechError::Input { why: why.into() }
}
#[cfg(test)]
mod tests;
