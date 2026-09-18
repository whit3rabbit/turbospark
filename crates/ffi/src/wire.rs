//! The JSON shapes that cross the boundary.
//!
//! **Options and results travel as JSON strings rather than as C structs, and
//! that is the single decision that keeps `turbospark.h` small.** It removes
//! about fifteen struct definitions and every layout, alignment and
//! versioning question with them, and it means adding a knob is a field here
//! rather than an ABI break. The cost is a serde round trip on calls that
//! happen once per session or once per second; the per-token path carries no
//! JSON at all and is a raw pointer and a length.
//!
//! Field names are `camelCase`, so a Swift `Codable` needs no
//! `CodingKeys` and the two sides cannot drift over a spelling.

use serde::{Deserialize, Serialize};

/// Reads a window or slot count that may be a number, `"auto"`, `null`, or
/// absent. `None` means automatic.
///
/// All three spellings of automatic are accepted because a Swift optional
/// bridges to an omitted key, a `null`, or the string `"auto"` depending on
/// how the caller writes its encoder, and making those mean different things
/// would be a trap invisible from the header. Any OTHER string is an error
/// rather than a silent fallback to automatic: `"atuo"` should be heard
/// about, not quietly honoured as the default.
pub fn sized(value: &Option<serde_json::Value>, name: &str) -> Result<Option<u32>, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) if s.eq_ignore_ascii_case("auto") => Ok(None),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .filter(|v| *v <= u32::MAX as u64)
            .map(|v| Some(v as u32))
            .ok_or_else(|| format!("{name} must be a non-negative integer, got {n}")),
        Some(other) => Err(format!("{name} must be a number or \"auto\", got {other}")),
    }
}

/// Reads an expert-cache slot policy: a COUNT, `"auto"`, `null`, or absent.
///
/// **THE ALLOWED-SET CHECK IS WHY THIS IS A FUNCTION RATHER THAN A `sized`
/// CALL AT EACH SITE.** `ExpertCacheSlots::Fixed` is built from whatever it
/// is handed and the setters panic outside `ALLOWED_CACHE_SLOTS`
/// (`crates/core` Gotcha 1), and this engine is linked INTO its host, so an
/// unvalidated count aborts the whole app rather than raising an error a GUI
/// can show. `ts_session_open` learned that once; every later caller reaches
/// the same check through here rather than restating it.
///
/// Absent, `null` and `"auto"` all mean `Auto`, matching [`sized`].
pub fn expert_cache_slots(
    value: &Option<serde_json::Value>,
) -> Result<model_io::ExpertCacheSlots, String> {
    match sized(value, "expertCacheSlots")? {
        None => Ok(model_io::ExpertCacheSlots::Auto),
        Some(n) => {
            if !foundation::runtime_config::ALLOWED_CACHE_SLOTS.contains(&n) {
                return Err(format!(
                    "expertCacheSlots must be \"auto\" or one of {:?}, got {n}",
                    foundation::runtime_config::ALLOWED_CACHE_SLOTS
                ));
            }
            Ok(model_io::ExpertCacheSlots::Fixed(n as usize))
        }
    }
}

/// Reads the routed-expert residency policy. Absent and `null` mean `auto`;
/// a typo is rejected before an install is opened.
pub fn expert_residency(value: &Option<String>) -> Result<model_io::ExpertResidency, String> {
    match value.as_deref() {
        None => Ok(model_io::ExpertResidency::Auto),
        Some(raw) => model_io::ExpertResidency::parse(raw).ok_or_else(|| {
            format!("expertResidency must be auto, streamed or mapped, got {raw:?}")
        }),
    }
}

/// Reads a load guard that may be a tier NAME, a byte ceiling, `null`, or
/// absent.
///
/// Absent and `null` mean `relaxed` for [`sized`]'s reason -- a Swift optional
/// bridges to either -- and, more importantly here, because `relaxed` is what
/// this binding did before the option existed. An unrecognized STRING is an
/// error rather than a silent fallback: quietly honouring `"strcit"` as the
/// default is exactly the trap `sized` refuses.
pub fn load_guard(value: &Option<serde_json::Value>) -> Result<model_io::LoadGuard, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(model_io::LoadGuard::default()),
        Some(serde_json::Value::String(s)) => model_io::LoadGuard::parse(&s.to_ascii_lowercase())
            .ok_or_else(|| {
                format!("loadGuard must be off, relaxed, balanced, strict or a byte count, got {s}")
            }),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .filter(|v| *v > 0)
            .map(|v| model_io::LoadGuard::Custom {
                max_counted_bytes: v,
            })
            .ok_or_else(|| format!("loadGuard as a byte ceiling must be positive, got {n}")),
        Some(other) => Err(format!(
            "loadGuard must be a tier name or a byte count, got {other}"
        )),
    }
}

/// Reads a `kvBits` selection: `"off"` (or `null`/absent), `"2"`, `"3"`,
/// `"3.5"`, or `"4"`.
///
/// Absent and `null` mean `off`, matching every other option here -- and,
/// more importantly, because `off` is what this binding did before the
/// option existed and what every frozen footprint row describes. An
/// unrecognized STRING is an error rather than a silent fallback, for
/// `load_guard`'s reason: quietly honouring a typo as `off` is the exact
/// trap this function exists to avoid, and unlike a fallback engine
/// PARAMETER, a caller who asked for quantization and silently got FP16
/// would measure the wrong memory footprint and believe it was the
/// requested one.
pub fn kv_bits(value: &Option<String>) -> Result<model_io::KvQuant, String> {
    match value {
        None => Ok(model_io::KvQuant::Off),
        Some(s) => model_io::KvQuant::parse(&s.to_ascii_lowercase())
            .map_err(|_| format!("kvBits must be off, 2, 3, 3.5 or 4, got {s:?}")),
    }
}

/// Arguments to `ts_recommend_json`. One field today, and a JSON blob rather
/// than a second `uint32_t` for the reason this crate takes every other
/// options bag as JSON: a knob added here is a field rather than an ABI
/// break, and the header stays readable.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RecommendOptions {
    /// Same spellings as `OpenOptions::load_guard`, and it MUST be the same
    /// value the host will open with; see `models::recommend_json`.
    pub load_guard: Option<serde_json::Value>,
    /// Same spellings as `OpenOptions::expert_cache_slots`, and it MUST be
    /// the same value the host will open with, for `load_guard`'s reason one
    /// term over: a footprint is `slots x layers x expert_stride`, so a
    /// ranking at one slot count and an `open()` at another describe
    /// different configurations rather than the same one approximately
    /// (`crates/catalog` Gotcha 9).
    pub expert_cache_slots: Option<serde_json::Value>,
}

/// Arguments to `ts_probe_json`. The same two knobs as [`RecommendOptions`]
/// plus the window, because a probe now reports a FIT and a fit is only
/// meaningful at a stated context and slot count.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ProbeOptions {
    pub context_window: Option<u32>,
    pub load_guard: Option<serde_json::Value>,
    pub expert_cache_slots: Option<serde_json::Value>,
}

/// Arguments to `ts_session_open`. Every field is optional; `{}` is valid and
/// means "everything automatic", which is what the two user-facing binaries
/// default to.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct OpenOptions {
    pub max_context: Option<serde_json::Value>,
    pub expert_cache_slots: Option<serde_json::Value>,
    /// `auto` | `streamed` | `mapped`. Absent means `auto`, which maps only
    /// when the minimum streamed cache cannot fit the measured headroom.
    pub expert_residency: Option<String>,
    /// `performance` | `balanced` | `efficiency`. Absent means ASK THE OS,
    /// which is what a user-facing binary should do (Low Power Mode selects
    /// `efficiency`) and what a measurement harness must not.
    pub power_profile: Option<String>,
    /// `off` | `relaxed` | `balanced` | `strict`, or a NUMBER, which is an
    /// absolute ceiling in bytes on what the engine may allocate. Absent
    /// means `relaxed`, which is what this binding did before the option
    /// existed -- and what every frozen footprint row describes.
    pub load_guard: Option<serde_json::Value>,
    /// Refuse to open when an AUTOMATIC window resolves below this. Says
    /// nothing about an explicit `maxContext`; see `model_io::LoadPolicy`.
    pub min_auto_context: Option<u32>,
    pub max_tokens_per_sec: Option<f64>,
    /// `"off"` | `"auto"` | a block size, as a number or a string. Absent
    /// means `auto`, which is what both other front ends default to.
    ///
    /// Resolved at OPEN and not per turn, because that is where the
    /// drafter's state is allocated and there is one engine per session --
    /// the same reason `turbospark-server` makes it a process-level flag.
    /// A named block that cannot be served is an ERROR from
    /// `ts_session_open`; `auto` that cannot be served opens fine and
    /// reports the reason in `sessionInfo.speculation`.
    pub speculation: Option<serde_json::Value>,
    /// `auto` | `mtp` | `dflash`. Absent means `auto`, which ENABLES an MTP
    /// head and only REPORTS a DFlash2 one (`docs/DFLASH2.md`).
    pub speculative_drafter: Option<String>,
    /// Path to a control vector (.gguf, llama.cpp layout) to steer with.
    pub steering: Option<String>,
    /// `ablate` | `add` | `clamp` | `renorm`. Default is `ablate` or whatever
    /// the vector declares.
    pub steering_mode: Option<String>,
    /// Multiplier on the edit strength (default 1.0; 0.0 is identity).
    pub steering_scale: Option<f64>,
    /// Layer range to steer, `START:END` inclusive 0-based (default all).
    pub steering_layers: Option<String>,
    /// Coefficient for `clamp` mode (default 0.0).
    pub steering_target: Option<f64>,
    /// Minimum activation magnitude to fire the edit (default 0.0).
    pub steering_gate: Option<f64>,
    /// Path to a standalone vision-tower sidecar install to attach to a
    /// text-only trunk (vision memory sidecar, Part A4). Absent or `null`
    /// means use the trunk's own tower, if it has one.
    pub vision_sidecar: Option<String>,
    /// `"off"` | `"2"` | `"3"` | `"3.5"` | `"4"`. Absent means `"off"`, which
    /// is what every release before this option existed produced byte for
    /// byte. An unsupported family or `head_dim` REFUSES `ts_session_open`
    /// by name rather than silently opening at FP16 -- see
    /// `docs/TRUBOQUANT.md`.
    pub kv_bits: Option<String>,
}

/// What a session resolved about directional steering, once, at open.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SteeringInfo {
    /// True when a control vector is active on this session.
    pub active: bool,
    /// **Whether this family's decode flow dispatches the edit AT ALL**, which
    /// is a different question from `active` and the one a caller has BEFORE
    /// it offers the control.
    ///
    /// `active` says a vector was loaded and is running. This says one COULD
    /// be: `open` refuses a direction set on a family that answers `false`
    /// (`runtime::steering::family_dispatches_steering`), so a GUI offering
    /// the knob against such an install offers one whose only outcome is a
    /// failed load. Reported for the same reason `vision.active` is: a control
    /// that can only fail is worse than a disabled one that says why.
    pub supported: bool,
    /// Why `supported` is false, in the exact words `open` would refuse with,
    /// or null when it is true.
    pub reason: Option<String>,
    /// `ablate` | `add` | `clamp` | `renorm`, present only when active.
    pub mode: Option<String>,
    /// Active scale multiplier, present only when active.
    pub scale: Option<f64>,
    /// Human-readable one-line description, or null when inactive.
    pub summary: Option<String>,
}

/// What a session resolved about this checkpoint's own tool-call markup.
///
/// **`native: false` IS NOT "TOOLS DO NOT WORK", and a caller that reads it
/// that way will hide the wrong control.** It says the checkpoint's own
/// framing hands no call over, so nothing arrives as a parsed call -- which is
/// exactly the case `docs/FORGE_GUARDRAILS.md`'s rescue exists for. Report it,
/// do not gate on it.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallingInfo {
    /// True when this dialect's markup carries calls this engine parses, i.e.
    /// when `StructuredAssistantDecoder` can emit one.
    pub native: bool,
    /// Why `native` is false, naming the dialect and (for Muse Glimmer) the
    /// markup that exists but has no parser. Null when `native` is true.
    pub reason: Option<String>,
}

/// What a session resolved about speculative decoding, once, at open.
///
/// Reported for the reason the resolved slot count is: a caller who asked
/// for `auto` named no number, and an install carrying a drafter that
/// decodes one token at a time with nothing said is the failure the feature
/// was built to end.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeculationInfo {
    /// How many tokens a round proposes, or null when this session does not
    /// speculate at all. Null is the "is it on" test; there is no separate
    /// flag to disagree with it.
    pub block: Option<usize>,
    /// `mtp` | `dflash`, present only when `block` is. Two drafters serve
    /// one family with different shapes and different measured optima, so a
    /// throughput figure is unreadable without knowing which one ran.
    pub drafter: Option<String>,
    /// Why speculation is off, when the caller might have expected it on.
    /// Null when they asked for `off` and got it, and null when it is on.
    pub reason: Option<String>,
}

/// Arguments to `ts_generate`.
///
/// The sampling defaults are the CLI's, so a GUI that sends `{}` gets what
/// `turbospark-check` gives with no flags.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerateOptions {
    pub max_new_tokens: u32,
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub stop_tokens: Vec<u32>,
    /// `off` | `low` | `medium` | `high` | `xhigh`. The ACCEPTED SET IS THE
    /// CHECKPOINT'S, not this crate's: a level the template rejects comes
    /// back as an error naming the level, because a per-family allowlist
    /// here would be a second, staler copy of a set the checkpoint already
    /// states.
    pub reasoning: String,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.2,
            top_k: 64,
            top_p: 0.95,
            repetition_penalty: 1.0,
            seed: None,
            stop: Vec::new(),
            stop_tokens: Vec::new(),
            reasoning: "off".to_string(),
        }
    }
}

/// Whether this session can accept an image, said once at open.
///
/// **`active` MEANS "AN IMAGE WOULD BE SERVED", NOT "A TOWER EXISTS".** A host
/// gates its attach control on this, so the two questions have to be the same
/// one: an install carrying a tower but no `preprocessor_config.json` refuses
/// every image by name (`crates/vision-io` Gotcha 6 -- the generic default is
/// wrong for this family by 16x and would silently produce a worse answer), and
/// a control that offered images anyway would promise work the loader then
/// declines. That is `swift/CLAUDE.md` Gotcha 23's rule: check the work exists
/// before adding the control that claims to do it.
///
/// `reason` is non-null exactly when a tower is present and `active` is false,
/// which is the only case a caller can act on.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct VisionInfo {
    pub active: bool,
    /// The `<|image_pad|>` id, or null when inactive. Reported rather than
    /// restated by a host, for `SessionInfo`'s standing reason.
    pub image_token_id: Option<i32>,
    pub reason: Option<String>,
    /// `"install"` when this session's tower (if any) comes from the trunk's
    /// own directory, `"sidecar"` when a `visionSidecar` option attached a
    /// standalone tower install instead (vision memory sidecar, Part A4).
    /// Null exactly when `active` is false and no tower is present at all --
    /// the same case that leaves `imageTokenId` null.
    pub source: Option<String>,
    /// The sidecar directory, present only when `source` is `"sidecar"`.
    pub sidecar_path: Option<String>,
    /// The RESOLVED pixel ceiling this session will actually preprocess an
    /// image against (vision memory sidecar, Part B3): the checkpoint's own
    /// declared `max_pixels`, or a smaller value this session's own
    /// `loadGuard` tier and committed memory could afford. Null exactly
    /// when `active` is false -- there is no ceiling to report for a
    /// session that cannot serve an image at all.
    pub max_pixels: Option<usize>,
}

/// One piece of a multimodal message's content, in prompt order.
///
/// **AN IMAGE CARRIES ITS PAYLOAD HERE AND ITS POSITION SOMEWHERE ELSE.**
/// `tokenizer::ContentPart::Image` is a bare placeholder by design (a template
/// renders one marker run and the tower's rows are injected later, by
/// position), so this type is where the path or the bytes live and the two
/// halves meet at the id sequence. Keep them ordered: the nth image pairs with
/// the nth marker run, and a container that reordered would pair each picture
/// with the wrong span.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum WirePart {
    Text {
        #[serde(default)]
        text: String,
    },
    /// Exactly one of `path` or `base64` -- both or neither is refused by
    /// [`WirePart::image_source`] rather than silently preferring one.
    Image {
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        base64: Option<String>,
    },
}

/// Where one image's bytes come from.
pub enum ImageSource<'a> {
    Path(&'a str),
    Base64(&'a str),
}

impl WirePart {
    /// The source of an image part, or an error naming what was wrong.
    ///
    /// Both spellings set is REFUSED rather than resolved by precedence: a
    /// caller that sent both meant one of them, and picking silently runs the
    /// wrong picture with no error anywhere.
    pub fn image_source(&self) -> Result<ImageSource<'_>, String> {
        let WirePart::Image { path, base64 } = self else {
            return Err("not an image part".to_string());
        };
        match (path.as_deref(), base64.as_deref()) {
            (Some(p), None) => Ok(ImageSource::Path(p)),
            (None, Some(b)) => Ok(ImageSource::Base64(b)),
            (Some(_), Some(_)) => Err(
                "an image part carries both \"path\" and \"base64\"; send exactly one".to_string(),
            ),
            (None, None) => Err(
                "an image part carries neither \"path\" nor \"base64\"; send exactly one"
                    .to_string(),
            ),
        }
    }
}

/// A message's content: a bare string, or ordered parts.
///
/// `untagged` so every caller that predates images sends and receives the
/// exact same JSON -- a plain string decodes to `Text` and re-serializes as a
/// plain string, which is what keeps `WindowFitOutcome.retained` byte-stable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum WireContent {
    Text(String),
    Parts(Vec<WirePart>),
}

impl Default for WireContent {
    fn default() -> Self {
        WireContent::Text(String::new())
    }
}

/// One chat message, in the shape `--messages-file` accepts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WireMessage {
    pub role: String,
    #[serde(default)]
    pub content: WireContent,
}

impl WireMessage {
    /// The TEXT-ONLY view, for the callers that summarize or count rather
    /// than render. Mirrors `tokenizer::Message::with_parts`'s own `content`.
    pub fn text(&self) -> String {
        match &self.content {
            WireContent::Text(t) => t.clone(),
            WireContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    WirePart::Text { text } => Some(text.as_str()),
                    WirePart::Image { .. } => None,
                })
                .collect(),
        }
    }

    /// This message's parts, or `None` when it is a plain string.
    pub fn parts(&self) -> Option<&[WirePart]> {
        match &self.content {
            WireContent::Text(_) => None,
            WireContent::Parts(parts) => Some(parts),
        }
    }

    /// This message's image parts, in order.
    pub fn image_parts(&self) -> impl Iterator<Item = &WirePart> {
        self.parts()
            .unwrap_or(&[])
            .iter()
            .filter(|p| matches!(p, WirePart::Image { .. }))
    }
}

/// The result of fitting a conversation into a context window budget.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct WindowFitOutcome {
    pub retained: Vec<WireMessage>,
    pub measured_tokens: u64,
    pub removed_turn_count: usize,
    pub has_room_for_generation: bool,
}

/// Special token identifiers for tokenizer introspection.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SpecialTokensInfo {
    pub bos_id: Option<i32>,
    pub eos_id: Option<i32>,
    pub pad_id: Option<i32>,
    pub end_of_turn_id: Option<i32>,
    pub stop_token_ids: Vec<i32>,
    pub think_start_id: Option<i32>,
    pub think_end_id: Option<i32>,
}

/// What `ts_generate` writes to its result out-parameter.
///
/// Carries the accumulated `content` and `reasoning` as well as streaming
/// them, so a caller that only wants the finished turn can ignore the
/// callback entirely.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResult {
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    /// How many of `promptTokens` continued from the previous turn's KV
    /// instead of being re-prefilled (`runtime::kv_prefix`, opted into
    /// unconditionally at `open`; see `open.rs`'s module doc). Zero on a
    /// session's first turn, on one where the render diverged anywhere
    /// (`/clear`-equivalent, an edited history, a re-tokenization that
    /// landed differently), or on a family with recurrent state or a
    /// sliding-window ring past its slack -- never an error, just a full
    /// prefill. Reported for the reason `crates/cli/src/chat.rs`'s
    /// `[prefix-reuse]` footer line exists: an integration that silently
    /// does nothing reads exactly like one that is working, and this field
    /// is what would have shown "0/33" instead of a working number looking
    /// plausible.
    pub reused_prefix_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    /// `endOfTurn` | `toolCalls` | `eos` | `stopString` | `maxTokens` |
    /// `cancelled`.
    pub stop_reason: String,
    /// Every tool call the model invoked this turn, in emission order, as
    /// `{"id", "name", "arguments"}` objects. Empty unless a call was
    /// parsed, which needs the tool to have been offered by name -- no
    /// `GenerateOptions` field offers tools yet, so this is always empty
    /// today and exists so the shape does not need a second pass when the
    /// binding grows a way to offer them. The same rows arrive on the
    /// stream as `TS_EVENT_TOOL` events.
    pub tool_calls: Vec<serde_json::Value>,
    /// Decode tokens per second, or null when no decoding happened (a run
    /// cancelled during prefill). Null rather than zero so a caller cannot
    /// plot a rate that was never measured.
    pub tokens_per_second: Option<f64>,
    pub content: String,
    pub reasoning: String,
    /// `normal` | `warn` | `critical`: the WORST memory pressure seen while
    /// this turn decoded.
    ///
    /// **`normal` when nothing was watching, which is the default.** The
    /// in-loop probe follows the power profile's stepping, so a session on
    /// `performance` reports the absence of a reading rather than a reading
    /// of "fine". This field exists to catch a SPIKE between polls;
    /// `ts_system_info_json` is what a status panel should read.
    pub peak_memory_pressure: String,
}

/// What `ts_session_info_json` returns: everything a status panel needs that
/// does not change during a session.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    pub model_path: String,
    pub family: String,
    /// The RESOLVED window, never what the caller asked for. Under `auto`
    /// the request carries no number, and the KV cache has already been
    /// allocated at this one.
    pub max_context: u32,
    /// The checkpoint's own trained context, or null for an install written
    /// before that was recorded. `maxContext` above it is legal and only
    /// degrades quality, which is why `pastTrainedContext` is reported
    /// rather than refused.
    pub trained_context: Option<u32>,
    pub past_trained_context: bool,
    /// The RESOLVED slot count. Worth 44.2 tok/s against 51.2 on one
    /// install, so no throughput or footprint figure is readable without it.
    pub expert_cache_slots: usize,
    /// The residency mode the session actually opened with.
    pub expert_residency: String,
    pub vocab_size: usize,
    pub dialect: String,
    /// `level` | `toggleOnly` | `none`. Says what KIND of control is
    /// meaningful; `reasoning_levels` below says what to put in it.
    pub reasoning_support: String,
    /// The levels this checkpoint's own template can express, ascending,
    /// always starting `"off"`. **BUILD THE MENU FROM THIS AND NOTHING
    /// ELSE.** The set is the checkpoint's and not derivable from the
    /// family: Qwen 3.8 answers `["off","low","medium","xhigh"]` and RAISES
    /// on `high`, where Harmony and Muse Glimmer answer
    /// `["off","low","medium","high"]`. Offering a level absent from here
    /// fails the turn with the template's own error.
    ///
    /// Levels rendering the same prompt are already collapsed, so a
    /// `toggleOnly` checkpoint answers exactly two and a `none` one answers
    /// `["off"]`. On `toggleOnly` the second entry is `"low"` BY POSITION
    /// and is not a label -- read `reasoning_support` and say "On".
    pub reasoning_levels: Vec<String>,
    pub steering: SteeringInfo,
    pub tool_calling: ToolCallingInfo,
    /// What speculative decoding resolved to. **`block` being non-null is a
    /// statement about this SESSION and not about the next turn**:
    /// acceptance is exact only at temperature 0, so a sampled turn decodes
    /// sequentially whatever this says.
    pub speculation: SpeculationInfo,
    pub vision: VisionInfo,
    pub special_tokens: SpecialTokensInfo,
    /// The RESOLVED `kvBits` selection: `"off"`, `"2"`, `"3"`, `"4"`, or
    /// `"3.5 (K3/V4)"` when the two widths differ. Not "requested" and
    /// "resolved" the way speculation is -- there is no auto-detect here,
    /// a named width either opens or the call fails, so what a caller asked
    /// for and what this session runs at are always the same value.
    pub kv_bits: String,
}

/// Arguments to `ts_server_start`. `{}` is valid and means "an OS-assigned
/// port, no auth" -- the same "everything automatic" convention
/// `OpenOptions` uses.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ServerOptions {
    /// Literal IPv4/IPv6 bind address; absent keeps loopback.
    pub host: Option<String>,
    /// Keep bounded HTTP body previews in memory for debugging.
    pub capture_text: bool,
    /// 0 (the default, and what an absent key also means) asks the OS for an
    /// ephemeral port; read the one actually bound back from
    /// `ts_server_info_json`.
    pub port: u16,
    /// Require this key on every request except `GET /health`, exactly as
    /// `turbospark-server --api-key` does. `None` (the default) leaves the
    /// server unauthenticated, appropriate for a server bound to loopback
    /// and reachable only by the process embedding it.
    pub api_key: Option<String>,
    /// `"on"` | `"off"`, or absent for the engine default (on).
    ///
    /// **PROCESS-LEVEL, matching `turbospark-server --guardrails` and for the
    /// same reason** (`crates/server/src/args.rs`): a per-request field would
    /// let any client opt its own traffic out of the repair this deployment
    /// chose. It is set here rather than per attached model because a host
    /// turning guardrails off is making a deployment decision, not a
    /// per-model one.
    ///
    /// This existing at all is a fix. `ts_server_start` took the
    /// `ChatModel::guardrails()` trait default and nothing could override it,
    /// so a host with its own guardrails setting had no way to make the
    /// SERVED path agree with it -- the setting looked global and reached
    /// only whatever the host itself did with a reply.
    pub guardrails: Option<String>,
    /// Optional embedding model (.safetensors directory or alias) to attach at startup.
    pub embedding_model: Option<String>,
    /// Optional Hugging Face mirror endpoint (e.g. https://hf-mirror.com).
    pub hf_endpoint: Option<String>,
    /// Deployment-wide default system prompt for requests that carry no system or
    /// developer message of their own (turbospark-server --system).
    pub default_system: Option<String>,
    /// Default reasoning effort when a request omits reasoning_effort
    /// (turbospark-server --reasoning).
    pub default_reasoning: Option<String>,
}

/// What `ts_server_info_json` returns.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    pub traffic: crate::server_transport::TrafficSnapshot,
    /// The port ACTUALLY bound, never the one requested: `port: 0` in
    /// `ServerOptions` asks for an OS-assigned one, so this is the only
    /// place that number is knowable.
    pub port: u16,
    /// The IP actually bound, read from the same `local_addr` call the port
    /// is, never the literal `server.rs` interpolated into its bind string.
    /// A caller that restates that literal is right until the bind changes
    /// and has no way to notice when it does.
    pub host: String,
    /// The FIRST attached model, or `""` when none is.
    ///
    /// Kept as a bare string for a reader written before a server could
    /// serve more than one; `models` is the answer that does not go stale.
    /// A host showing this alone on a two-model server is showing half the
    /// truth, which is why the Swift binding surfaces both.
    pub model_id: String,
    /// Every attached canonical model id, in attachment order.
    ///
    /// `GET /v1/models` expands each generative backend with its
    /// `claude-turbospark-<canonical-id>` discovery alias. Those aliases are
    /// intentionally absent here, so a host can keep using the id returned
    /// by attach to identify and detach its session.
    pub models: Vec<String>,
    pub auth_enabled: bool,
    /// Seconds since `ts_server_start` returned. From a monotonic clock, so
    /// it is unaffected by the wall clock moving under a long-running host.
    pub uptime_seconds: u64,
}

/// What `ts_server_poll_events_json` returns.
///
/// **`dropped` IS PART OF THE PAYLOAD RATHER THAN A SEPARATE QUERY**, because
/// a gap only means anything beside the events it interrupts. A host that
/// had to ask twice could report the drop against the wrong window, and one
/// that never asked would render a lossy log as a complete one -- which
/// looks exactly like a server that was idle.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerEvents {
    pub events: Vec<turbospark_server::observe::ServerEvent>,
    /// Events discarded since the previous poll, oldest first. Zero on a
    /// host that keeps up, which is every host polling on a timer.
    pub dropped: u64,
}

/// What `ts_session_phases_json` returns: `TURBOSPARK_PHASES=1`'s breakdown.
///
/// **Cumulative over every forward pass the runner has served, PREFILL
/// INCLUDED**, so a per-token number here is an average over the session's
/// whole context range rather than a number at the current context. And the
/// buckets cover the inside of `produce` only: the sampler and the
/// detokenizer run after it returns and appear in none of them.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseReport {
    pub calls: u64,
    pub total_ms_per_call: f64,
    pub gpu_wait_ms: f64,
    pub final_wait_ms: f64,
    pub router_ms: f64,
    pub expert_io_ms: f64,
    pub bind_ms: f64,
    pub pipeline_wait_ms: f64,
    pub cb1_gpu_ms: f64,
    pub routed_cb_gpu_ms: f64,
    pub final_cb_gpu_ms: f64,
    pub expert_requests: u64,
    pub expert_hits: u64,
    /// Null when nothing has been requested yet, rather than a 0% hit rate
    /// on no data.
    pub expert_hit_rate: Option<f64>,
}
