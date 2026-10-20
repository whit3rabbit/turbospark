//! The opt-in ANE+Steel prefill split for the dense qwen family. Compiled
//! only with the `ane-prefill` feature and taken only when
//! `TURBOSPARK_ANE_PREFILL` (or [`RealForwardRunner::set_ane_prefill`]) asks
//! for it. Steel stays the default. On Qwen3.8-27B (M4 Max, a 2,940-token
//! prompt at width 2048) it measured 1.347x prefill at 0.646x modeled energy
//! with greedy output unchanged (`docs/SPEEDUP_PLAYBOOK.md`,
//! `docs/MOE_PRIOR_ART.md`).
//!
//! It splits dense projections by output rows inside the chunked prefill.
//! One Core ML prediction runs rows `[0, MA)` of a class's matrices on the
//! ANE while Steel computes rows `[MA, rows)` straight into the shared
//! output. The classes, each one split point per layer it appears in
//! (Qwen3.8-27B shapes):
//!
//! - `qg`: attention q with its gate (12288 x 5120), mask-1 layers. K and V
//!   (1024 rows each) stay on Steel, in the same buffer as Steel's q rows.
//! - `qkvz`: GDN qkv (10240 x 5120) and z (6144 x 5120) in one prediction,
//!   mask-2 layers. a and b (48 rows, the scalar kernel) stay on the GPU,
//!   in the same buffer as Steel's rows.
//! - `o`: o_proj or out_proj (5120 x 6144), every layer.
//! - `gateup`: MLP gate and up (17408 x 5120) in one prediction.
//! - `down`: MLP down (5120 x 17408).
//!
//! The default arm encodes a micro-batch as ONE command buffer. Core ML is
//! driven from the host, so the buffer has to break at every split point p:
//!
//! ```text
//! seg(p)   [GPU waits for E >= base + p] convert(p-1), the layer up to
//!          point p's input, prep(p): weights to F16, scaled activations
//! rows(p)  Steel rows [MA, rows) of point p's matrices, plus the class's
//!          Steel-only projections (K/V, a/b)
//! host     wait seg(p) -> Core ML predict(p) -> signal E = base + p + 1
//! ```
//!
//! Every buffer is encoded before any is committed. A class with no program
//! for the micro-batch's token count encodes exactly as the default arm
//! does, in the running segment.
//!
//! Programs and shares come from a directory that
//! `crates/coreml/tools/gen_ane_prefill_programs.py` writes: Core ML
//! packages plus a `manifest.json` naming, per class and token count, each
//! matrix's ANE rows (`ane_rows`, balanced per shape) and weight-plane rows
//! (`plane_rows`, the largest ANE rows of that matrix over the token counts,
//! so one IOSurface weight plane serves every count). A micro-batch of m
//! rows takes the smallest count N >= m, its activations padded, and splits
//! only when m is at least half the smallest N.
//!
//! Failure handling. Programs that are missing, fail to load, or do not
//! match the install turn the split off for the runner with one warning, and
//! prefill runs on Steel. A split micro-batch snapshots the GDN recurrent
//! state before it commits; a failed prediction, or any non-finite ANE row,
//! restores the snapshot and redoes the micro-batch on Steel. K/V slot
//! writes are idempotent and the KV cursor advances only after success, so
//! nothing else needs undoing.
//!
//! The split steps aside (and the micro-batch stays 64 rows) while an MTP
//! head or DFlash2 drafter is open, because their priming state is sized for
//! 64-row micro-batches. Image micro-batches run Steel.
//!
//! Env:
//! - `TURBOSPARK_ANE_PREFILL`: `1` (or `on`) splits. `break` keeps every
//!   buffer break and host handoff but Steel takes every row, a diagnostic
//!   for the cost of the breaks alone. Anything else is off.
//! - `TURBOSPARK_ANE_PREFILL_DIR`: the programs directory.
//! - `TURBOSPARK_ANE_PREFILL_STATS=1`: one counter line per split
//!   micro-batch on stderr.
//! - `TURBOSPARK_ANE_PREFILL_FAULT=P`: a diagnostic. The prediction at point
//!   P of the first split micro-batch that reaches it fails, once per
//!   process, so the Steel redo can be checked end to end.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use model_io::{ArchConfig, ResidentIndex};
use turbospark_coreml::{ComputeUnits, CoreMlModel, LoadOptions, SharedF16Plane};

use super::batched::BatchedScratch;
use super::batched_layers::{
    encode_dense_ffn_act_batched, encode_dense_ffn_residual_batched,
    encode_full_attention_core_batched, encode_full_attention_in_proj_batched,
    encode_linear_core_batched, encode_linear_in_proj_batched, full_attention_kv_slots,
};
use super::prefill_layers::{
    encode_qwen_dense_attn_residual_batched, encode_qwen_dense_input_norm_batched,
    encode_qwen_dense_layer_batched_post_ffn,
};
use super::{layer_tensor, QkNormConvention, RealQwenState, TRUNK_PREFIX};
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::encode_gemm_rows;
use crate::real_forward_types::{DecodeScratch, RealForwardError, QWEN_STEEL_PREFILL_BATCH};
use crate::real_forward_utils::{norm_view, resident_matrix};

use gpu::ane_split as ane;

/// The micro-batch width the split runs at. The batched scratch takes it
/// while the split is on, about 674 MiB at Qwen3.8-27B's shape.
pub(crate) const ANE_PREFILL_WIDTH: usize = 2048;

/// The manifest's format version this build reads.
const MANIFEST_FORMAT: u64 = 1;

/// Which prefill arm [`AnePrefill`] asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnePrefillMode {
    /// Steel only, the default.
    Off,
    /// Automatically enabled on Apple Silicon M4+ for dense Qwen models.
    Auto,
    /// Split each class between the ANE and Steel.
    Split,
    /// Break the micro-batch at every split point but run every row on
    /// Steel: the breaks' cost alone, a diagnostic.
    Break,
}

impl AnePrefillMode {
    pub fn from_env_value(v: &str) -> Self {
        match v.trim().to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "1" | "on" | "split" => Self::Split,
            "break" => Self::Break,
            _ => Self::Off,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "off" | "0" | "false" => Some(Self::Off),
            "1" | "on" | "true" | "split" => Some(Self::Split),
            "break" => Some(Self::Break),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Split => "split",
            Self::Break => "break",
        }
    }

    pub fn is_split(&self) -> bool {
        matches!(self, Self::Split)
    }
}

/// Whether this chip brand or device name indicates an Apple Silicon M4 or newer chip
/// (e.g. "Apple M4", "Apple M4 Pro", "Apple M4 Max", "Apple M5", or A18+).
pub fn chip_is_m4_or_newer(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if let Some(pos) = lower.find("apple m") {
        let rest = &lower[pos + "apple m".len()..];
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(gen) = num_str.parse::<u32>() {
            return gen >= 4;
        }
    }
    for token in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        if token.starts_with('m') && token.len() >= 2 {
            if let Ok(gen) = token[1..].parse::<u32>() {
                return gen >= 4;
            }
        }
        if token.starts_with('a') && token.len() >= 3 {
            if let Ok(gen) = token[1..].parse::<u32>() {
                return gen >= 18;
            }
        }
    }
    false
}

/// Checks whether the host hardware has an Apple Silicon M4 or newer chip.
pub fn hardware_supports_ane_prefill() -> bool {
    if let Some((_, name)) = crate::recommended_max_working_set() {
        if chip_is_m4_or_newer(&name) {
            return true;
        }
    }
    false
}

/// The runner's split configuration.
#[derive(Clone, Debug)]
pub(crate) struct AnePrefill {
    mode: AnePrefillMode,
    programs: Option<PathBuf>,
    /// Set while a failed split micro-batch is redone on Steel.
    redo: bool,
    warned_chunk: bool,
}

impl AnePrefill {
    pub(crate) fn from_env() -> Self {
        Self::new(
            AnePrefillMode::from_env_value(
                &std::env::var("TURBOSPARK_ANE_PREFILL").unwrap_or_default(),
            ),
            std::env::var_os("TURBOSPARK_ANE_PREFILL_DIR").map(PathBuf::from),
        )
    }

    pub(crate) fn new(mode: AnePrefillMode, programs: Option<PathBuf>) -> Self {
        Self {
            mode,
            programs,
            redo: false,
            warned_chunk: false,
        }
    }

    pub(crate) fn mode(&self) -> AnePrefillMode {
        self.mode
    }

    pub(crate) fn programs(&self) -> Option<&Path> {
        self.programs.as_deref()
    }

    fn on(&self) -> bool {
        match self.mode {
            AnePrefillMode::Off => false,
            AnePrefillMode::Auto => hardware_supports_ane_prefill(),
            AnePrefillMode::Split | AnePrefillMode::Break => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitMode {
    Ane,
    Break,
}

/// A projection class, one split point per layer it appears in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Qg,
    Qkvz,
    O,
    GateUp,
    Down,
}

impl Class {
    const ALL: [Class; 5] = [Class::Qg, Class::Qkvz, Class::O, Class::GateUp, Class::Down];

    fn name(self) -> &'static str {
        match self {
            Class::Qg => "qg",
            Class::Qkvz => "qkvz",
            Class::O => "o",
            Class::GateUp => "gateup",
            Class::Down => "down",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name)
    }

    /// Whether this class has a split point in a layer of this kind.
    fn in_layer(self, linear: bool) -> bool {
        match self {
            Class::Qg => !linear,
            Class::Qkvz => linear,
            _ => true,
        }
    }
}

/// The projection widths the classes are built from.
#[derive(Clone, Copy, Debug)]
struct Dims {
    hidden: usize,
    inter: usize,
    q_dim: usize,
    qkv_dim: usize,
    value_dim: usize,
}

impl Dims {
    fn new(arch: &ArchConfig, shape: gpu::GdnShape) -> Self {
        Self {
            hidden: arch.hidden_size as usize,
            inter: arch.intermediate_size as usize,
            q_dim: (arch.num_heads * arch.full_head_dim) as usize,
            qkv_dim: shape.qkv_dim() as usize,
            value_dim: shape.value_dim() as usize,
        }
    }

    /// `class`'s input width and each of its matrices' output rows.
    fn class(self, class: Class) -> (usize, Vec<usize>) {
        match class {
            Class::Qg => (self.hidden, vec![2 * self.q_dim]),
            Class::Qkvz => (self.hidden, vec![self.qkv_dim, self.value_dim]),
            Class::O => (self.q_dim, vec![self.hidden]),
            Class::GateUp => (self.hidden, vec![self.inter, self.inter]),
            Class::Down => (self.inter, vec![self.hidden]),
        }
    }
}

/// One manifest program, validated against the install.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    class: Class,
    tokens: usize,
    k: usize,
    /// Per matrix: ANE rows, weight-plane rows.
    mats: Vec<(usize, usize)>,
    package: String,
}

#[derive(serde::Deserialize)]
struct ManifestFile {
    format: u64,
    programs: Vec<ManifestProgram>,
}

#[derive(serde::Deserialize)]
struct ManifestProgram {
    class: String,
    k: usize,
    tokens: usize,
    matrices: Vec<ManifestMatrix>,
    package: String,
}

#[derive(serde::Deserialize)]
struct ManifestMatrix {
    rows: usize,
    ane_rows: usize,
    plane_rows: usize,
}

/// Parses a `manifest.json` and checks every program against this install's
/// shapes. Any mismatch refuses the whole manifest: a split built for
/// another model would write rows at the wrong offsets.
fn parse_manifest(text: &str, dims: Dims) -> Result<Vec<Entry>, String> {
    let file: ManifestFile =
        serde_json::from_str(text).map_err(|e| format!("manifest.json does not parse: {e}"))?;
    if file.format != MANIFEST_FORMAT {
        return Err(format!(
            "manifest.json is format {}, this build reads {MANIFEST_FORMAT}",
            file.format
        ));
    }
    if dims.q_dim != dims.value_dim && file.programs.iter().any(|p| p.class == "o") {
        return Err(format!(
            "class o needs o_proj and out_proj to share K, but they are {} and {}",
            dims.q_dim, dims.value_dim
        ));
    }
    let mut entries: Vec<Entry> = Vec::new();
    for p in file.programs {
        let class = Class::from_name(&p.class)
            .ok_or_else(|| format!("unknown class {:?} in manifest.json", p.class))?;
        let (k, rows) = dims.class(class);
        let what = format!("{} at {} tokens", class.name(), p.tokens);
        if p.k != k || p.matrices.len() != rows.len() {
            return Err(format!(
                "{what}: manifest K {} with {} matrices, the install needs K {k} with {}",
                p.k,
                p.matrices.len(),
                rows.len()
            ));
        }
        if p.tokens == 0 || p.tokens > ANE_PREFILL_WIDTH {
            return Err(format!(
                "{what}: token count outside 1..={ANE_PREFILL_WIDTH}"
            ));
        }
        let mut mats = Vec::with_capacity(rows.len());
        for (m, &r) in p.matrices.iter().zip(&rows) {
            let ok = m.rows == r
                && m.ane_rows > 0
                && m.ane_rows < r
                && m.ane_rows % 32 == 0
                && m.plane_rows >= m.ane_rows;
            if !ok {
                return Err(format!(
                    "{what}: matrix of {} rows with {} ANE rows on a {}-row plane, the install \
                     has {r} rows and Steel needs a multiple of 32 inside (0, {r})",
                    m.rows, m.ane_rows, m.plane_rows
                ));
            }
            mats.push((m.ane_rows, m.plane_rows));
        }
        if p.package.contains('/') || p.package.contains("..") {
            return Err(format!(
                "{what}: package {:?} is not a plain name",
                p.package
            ));
        }
        if entries
            .iter()
            .any(|e| e.class == class && e.tokens == p.tokens)
        {
            return Err(format!("{what} is listed twice"));
        }
        entries.push(Entry {
            class,
            tokens: p.tokens,
            k,
            mats,
            package: p.package,
        });
    }
    if entries.is_empty() {
        return Err("manifest.json lists no programs".into());
    }
    Ok(entries)
}

/// The token count a micro-batch of `m` rows runs at, if any: the smallest
/// listed count >= m, when m is at least half the smallest one.
fn slot_for(slots: &[usize], m: usize) -> Option<usize> {
    let floor = slots.first()?.div_ceil(2);
    slots.iter().copied().find(|&n| n >= m && m >= floor)
}

fn coreml_err(e: turbospark_coreml::CoreMlError) -> String {
    format!("Core ML: {e}")
}

/// An IOSurface plane and the no-copy Metal buffer over it. The buffer is
/// declared first so it drops before the memory it wraps.
struct Plane {
    buffer: gpu::MetalBuffer,
    plane: SharedF16Plane,
}

impl Plane {
    fn new(context: &gpu::MetalContext, rows: usize, cols: usize) -> Result<Self, String> {
        let plane = SharedF16Plane::new(rows, cols).map_err(coreml_err)?;
        let buffer =
            gpu::wrap_page_aligned_no_copy(context.device(), plane.base_ptr(), plane.alloc_size())
                .map_err(|e| format!("Metal: {e}"))?;
        Ok(Self { buffer, plane })
    }
}

/// What a plane holds for its program: weight input `i`, the activations,
/// or output `i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlaneKind {
    W(usize),
    A,
    O(usize),
}

/// Every plane, shared by shape. Split points run one at a time: point
/// p + 1's prep waits for p's prediction, and its prediction waits for the
/// segment that converted p's output. So any two points may share a plane
/// of the same kind and shape, and only the two inputs (or outputs) of one
/// program need distinct ones.
#[derive(Default)]
struct Planes {
    all: Vec<(PlaneKind, usize, usize, Plane)>,
}

impl Planes {
    fn get_or_new(
        &mut self,
        context: &gpu::MetalContext,
        kind: PlaneKind,
        rows: usize,
        cols: usize,
    ) -> Result<usize, String> {
        if let Some(i) = self
            .all
            .iter()
            .position(|(k, r, c, _)| *k == kind && *r == rows && *c == cols)
        {
            return Ok(i);
        }
        self.all
            .push((kind, rows, cols, Plane::new(context, rows, cols)?));
        Ok(self.all.len() - 1)
    }

    fn get(&self, i: usize) -> &Plane {
        &self.all[i].3
    }

    fn bytes(&self) -> usize {
        self.all.iter().map(|(.., p)| p.plane.alloc_size()).sum()
    }
}

/// One loaded program: a class at one token count, its matrices' ANE rows,
/// and the planes it binds.
struct Program {
    class: Class,
    tokens: usize,
    mas: Vec<usize>,
    w: Vec<usize>,
    a: usize,
    o: Vec<usize>,
    model: CoreMlModel,
}

fn predict(p: &Program, planes: &Planes) -> Result<(), String> {
    const W: [&str; 2] = ["w0", "w1"];
    const O: [&str; 2] = ["o0", "o1"];
    let mut inputs: Vec<(&str, &SharedF16Plane)> = vec![("a", &planes.get(p.a).plane)];
    inputs.extend(
        p.w.iter()
            .enumerate()
            .map(|(i, &w)| (W[i], &planes.get(w).plane)),
    );
    let outputs: Vec<(&str, &SharedF16Plane)> =
        p.o.iter()
            .enumerate()
            .map(|(i, &o)| (O[i], &planes.get(o).plane))
            .collect();
    p.model
        .predict_shared(&inputs, &outputs)
        .map_err(coreml_err)
}

pub(crate) struct AneSplitState {
    event: ane::HostEvent,
    /// Per-token inverse activation scales, written by each point's prep and
    /// read by its convert.
    inv: gpu::MetalBuffer,
    dir: PathBuf,
    /// Why the split is off for this runner, once loading failed.
    off: Option<String>,
    entries: Vec<Entry>,
    slots: Vec<usize>,
    planes: Planes,
    /// Loaded on the first split micro-batch; `break` needs none.
    programs: Vec<Program>,
}

impl AneSplitState {
    /// Reads and checks the manifest. Programs load later, on the first
    /// split micro-batch.
    fn open(context: &gpu::MetalContext, dir: &Path, dims: Dims) -> Self {
        let path = dir.join("manifest.json");
        let parsed = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))
            .and_then(|text| parse_manifest(&text, dims));
        match parsed {
            Ok(entries) => Self::new(context, dir, entries, None),
            Err(reason) => Self::new(context, dir, Vec::new(), Some(reason)),
        }
    }

    fn new(
        context: &gpu::MetalContext,
        dir: &Path,
        entries: Vec<Entry>,
        off: Option<String>,
    ) -> Self {
        let mut slots: Vec<usize> = entries.iter().map(|e| e.tokens).collect();
        slots.sort_unstable();
        slots.dedup();
        Self {
            event: ane::HostEvent::new(context),
            inv: context.new_output_buffer((ANE_PREFILL_WIDTH * 4) as u64),
            dir: dir.to_path_buf(),
            off,
            entries,
            slots,
            planes: Planes::default(),
            programs: Vec::new(),
        }
    }

    /// Opens every manifest program and binds its planes, paying each one's
    /// Neural Engine load with two warm-up predictions.
    fn load_programs(&mut self, context: &gpu::MetalContext) -> Result<(), String> {
        let started = Instant::now();
        let cache = self.dir.join("compiled");
        for e in self.entries.clone() {
            let model = CoreMlModel::open(
                &self.dir.join(&e.package),
                &cache,
                &LoadOptions::new(ComputeUnits::CpuAndNeuralEngine),
            )
            .map_err(|err| format!("{}: {}", e.package, coreml_err(err)))?;
            let mut w = Vec::new();
            let mut o = Vec::new();
            for (i, &(ma, wr)) in e.mats.iter().enumerate() {
                w.push(self.planes.get_or_new(context, PlaneKind::W(i), wr, e.k)?);
                o.push(
                    self.planes
                        .get_or_new(context, PlaneKind::O(i), ma, e.tokens)?,
                );
            }
            let a = self
                .planes
                .get_or_new(context, PlaneKind::A, e.tokens, e.k)?;
            let program = Program {
                class: e.class,
                tokens: e.tokens,
                mas: e.mats.iter().map(|&(ma, _)| ma).collect(),
                w,
                a,
                o,
                model,
            };
            for _ in 0..2 {
                predict(&program, &self.planes).map_err(|err| format!("{}: {err}", e.package))?;
            }
            self.programs.push(program);
        }
        eprintln!(
            "[ane-prefill] {} programs ready in {:.2} s, {:.0} MiB of planes",
            self.programs.len(),
            started.elapsed().as_secs_f64(),
            self.planes.bytes() as f64 / (1 << 20) as f64
        );
        Ok(())
    }

    fn turn_off(&mut self, reason: String) {
        eprintln!("[ane-prefill] off: {reason}. Prefill runs on Steel.");
        self.programs.clear();
        self.planes = Planes::default();
        self.off = Some(reason);
    }
}

/// Makes `state` ready for a micro-batch of `m` rows in `mode` from `dir`,
/// reading the manifest when the directory changed and loading the programs
/// the first time a split micro-batch has a program slot. False when this
/// micro-batch does not split, or the split is off for this runner.
fn ensure(
    state: &mut Option<AneSplitState>,
    context: &gpu::MetalContext,
    dims: Dims,
    dir: Option<&Path>,
    mode: SplitMode,
    m: usize,
) -> bool {
    let candidate = dir.map(Path::to_path_buf).or_else(|| {
        if let Some(env_dir) = std::env::var_os("TURBOSPARK_ANE_PREFILL_DIR").map(PathBuf::from) {
            if env_dir.join("manifest.json").exists() {
                return Some(env_dir);
            }
        }
        if let Ok(home) = std::env::var("HOME") {
            let p1 = PathBuf::from(&home).join(".turbospark/models/ane-prefill/qwen38");
            if p1.join("manifest.json").exists() {
                return Some(p1);
            }
            let p2 = PathBuf::from(&home).join(".turbospark/ane-prefill/qwen38");
            if p2.join("manifest.json").exists() {
                return Some(p2);
            }
            let p3 = PathBuf::from(&home).join("models/ane-prefill/qwen38");
            if p3.join("manifest.json").exists() {
                return Some(p3);
            }
        }
        None
    });
    let Some(dir) = candidate.as_deref().or(dir) else {
        if state.is_none() {
            let reason = "TURBOSPARK_ANE_PREFILL_DIR is not set and no ane-prefill programs found"
                .to_string();
            eprintln!("[ane-prefill] off: {reason}. Prefill runs on Steel.");
            *state = Some(AneSplitState::new(
                context,
                Path::new(""),
                Vec::new(),
                Some(reason),
            ));
        }
        return false;
    };
    if state.as_ref().is_none_or(|s| s.dir != dir) {
        let st = AneSplitState::open(context, dir, dims);
        if let Some(reason) = &st.off {
            eprintln!("[ane-prefill] off: {reason}. Prefill runs on Steel.");
        }
        *state = Some(st);
    }
    let st = state.as_mut().expect("set above");
    if st.off.is_some() || slot_for(&st.slots, m).is_none() {
        return false;
    }
    if mode == SplitMode::Ane && st.programs.is_empty() {
        if let Err(reason) = st.load_programs(context) {
            st.turn_off(reason);
            return false;
        }
    }
    true
}

/// The diagnostic fault point, once per process.
fn take_fault_point() -> Option<usize> {
    static FIRED: AtomicBool = AtomicBool::new(false);
    let p = std::env::var("TURBOSPARK_ANE_PREFILL_FAULT")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (!FIRED.swap(true, Ordering::Relaxed)).then_some(p)
}

impl RealForwardRunner {
    /// The qwen dense driver's Steel micro-batch for the next chunk:
    /// [`ANE_PREFILL_WIDTH`] while the split is on and no drafter is open,
    /// [`QWEN_STEEL_PREFILL_BATCH`] otherwise. Widens `scratch.x` to match,
    /// and warns once when the caller's chunk is too small for the split to
    /// engage.
    pub(crate) fn ane_prefill_steel_rows(&mut self, chunk_tokens: usize) -> usize {
        if !self.ane_prefill.on() || self.real_mtp.is_some() || self.real_dflash.is_some() {
            return QWEN_STEEL_PREFILL_BATCH;
        }
        let hidden = self.arch.hidden_size as usize;
        if (self.scratch.x.length() as usize) < hidden * ANE_PREFILL_WIDTH * 2 {
            self.scratch.x = self
                .context
                .new_output_buffer((hidden * ANE_PREFILL_WIDTH * 2) as u64);
        }
        if !self.prefill_chunk_is_final && chunk_tokens < 512 && !self.ane_prefill.warned_chunk {
            eprintln!(
                "[ane-prefill] a {chunk_tokens}-token prefill chunk is too small for the split, \
                 which needs micro-batches of at least 512 rows. Use a prefill chunk of 2048."
            );
            self.ane_prefill.warned_chunk = true;
        }
        ANE_PREFILL_WIDTH
    }

    /// Whether a qwen dense micro-batch of `m` rows on the batched arm takes
    /// the split, and in which mode. Loads the programs the first time.
    fn ane_prefill_mode_for(&mut self, m: usize, batched: bool) -> Option<SplitMode> {
        let mode = match self.ane_prefill.mode {
            AnePrefillMode::Off => return None,
            AnePrefillMode::Auto => {
                if hardware_supports_ane_prefill() {
                    SplitMode::Ane
                } else {
                    return None;
                }
            }
            AnePrefillMode::Split => SplitMode::Ane,
            AnePrefillMode::Break => SplitMode::Break,
        };
        if self.ane_prefill.redo
            || !batched
            || self.prompt_vision.is_some()
            || self.real_mtp.is_some()
            || self.real_dflash.is_some()
        {
            return None;
        }
        let qwen = self.real_qwen.as_mut()?;
        if !qwen.batched_prefill().steel_gemm || qwen.batched_prefill().batch < m {
            return None;
        }
        let dims = Dims::new(&self.arch, qwen.shape);
        let dir = self.ane_prefill.programs.as_deref();
        ensure(&mut qwen.ane_split, &self.context, dims, dir, mode, m).then_some(mode)
    }

    /// The split arm of a qwen dense micro-batch, when it applies. `first`
    /// already holds the embedding lookups, and comes back unused when the
    /// micro-batch should run the default arm.
    pub(crate) fn try_ane_prefill_micro_batch(
        &mut self,
        first: gpu::PassEncoder,
        tokens: &[i32],
        start_position: usize,
        want_head: bool,
        batched: bool,
        logits: &mut [foundation::LogitValue],
    ) -> AneMicroBatch {
        let m = tokens.len();
        let Some(mode) = self.ane_prefill_mode_for(m, batched) else {
            return AneMicroBatch::Declined(first);
        };
        let arch = self.arch.clone();
        let hidden = arch.hidden_size as usize;
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let embed_name = "language_model.model.embed_tokens.weight";
        let (context, weights, index, scratch, kv, qwen, resid_capture, steering) = (
            &mut self.context,
            &self.weights,
            &self.index,
            &self.scratch,
            &mut self.kv,
            self.real_qwen.as_ref().expect("checked dense above"),
            self.resid_capture.as_ref(),
            self.steering.as_ref(),
        );
        let outcome = run_micro_batch(
            context,
            first,
            weights,
            index,
            scratch,
            kv,
            qwen,
            &arch,
            resid_capture,
            steering,
            mode,
            use_silu,
            start_position,
            m,
            |context, pass| {
                if want_head {
                    super::prefill_layers::encode_qwen_dense_chunk_head(
                        context, pass, weights, index, scratch, qwen, &arch, embed_name, hidden,
                        vocab, m,
                    )
                } else {
                    pass.relabel("qwen split segment (no head)");
                    Ok(())
                }
            },
        );
        AneMicroBatch::Ran(match outcome {
            Err(e) => Err(e),
            Ok(SplitOutcome::Done(times)) => {
                self.phases.final_cb_gpu_nanos += (times.gpu_busy * 1e9) as u64;
                self.phases.final_wait_nanos += (times.wall * 1e9) as u64;
                self.finish_qwen_dense_micro_batch(start_position, m, want_head, logits)
            }
            Ok(SplitOutcome::Redo { snapshot, reason }) => {
                eprintln!(
                    "[ane-prefill] {reason} in the micro-batch at {start_position}; redoing it on \
                     Steel"
                );
                let queue = self.context.queue();
                self.real_qwen
                    .as_mut()
                    .expect("checked dense above")
                    .gdn
                    .restore(&snapshot, queue);
                drop(snapshot);
                self.ane_prefill.redo = true;
                let redone = self.prefill_micro_batch_qwen_dense_inner(
                    tokens,
                    start_position,
                    want_head,
                    batched,
                    logits,
                );
                self.ane_prefill.redo = false;
                redone
            }
        })
    }
}

/// What [`RealForwardRunner::try_ane_prefill_micro_batch`] did.
pub(crate) enum AneMicroBatch {
    /// The split ran the micro-batch (redone on Steel after a failure, if
    /// one happened).
    Ran(Result<(), RealForwardError>),
    /// The split does not apply: the default arm takes the pass.
    Declined(gpu::PassEncoder),
}

/// What one split micro-batch cost, in seconds.
struct SplitTimes {
    /// GPU busy time: each buffer's GPU interval, with the stretch an
    /// event-gated buffer spent waiting for the host left out.
    gpu_busy: f64,
    /// First commit to the last buffer's completion, on the host.
    wall: f64,
}

enum SplitOutcome {
    Done(SplitTimes),
    /// The micro-batch must be redone on Steel from `snapshot`.
    Redo {
        snapshot: gpu::GdnSnapshot,
        reason: String,
    },
}

/// One matrix of a split point: its tensor, output rows and destination.
struct Proj<'b> {
    name: String,
    rows: usize,
    out: &'b gpu::MetalBuffer,
}

/// The micro-batch's buffers as they are encoded.
struct Split<'a> {
    weights: &'a gpu::ResidentGpuWeights,
    index: &'a ResidentIndex,
    batched: &'a BatchedScratch,
    state: &'a AneSplitState,
    /// Each class split in this micro-batch, with its program when the mode
    /// is [`SplitMode::Ane`] (`None` for [`SplitMode::Break`]).
    classes: Vec<(Class, Option<usize>)>,
    nonfinite: gpu::MetalBuffer,
    m: usize,
    base: u64,
    pass: gpu::PassEncoder,
    segs: Vec<gpu::PassEncoder>,
    rows: Vec<gpu::PassEncoder>,
    points: Vec<(Class, Option<usize>)>,
}

impl Split<'_> {
    /// Every row of `projs` on Steel, exactly the default arm's call.
    fn full(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        x: &gpu::MetalBuffer,
        cols: usize,
        projs: &[Proj<'_>],
    ) -> Result<(), RealForwardError> {
        for p in projs {
            encode_gemm_rows(
                context,
                pass,
                self.weights,
                self.index,
                &p.name,
                p.rows,
                cols,
                (x, 0),
                (p.out, 0),
                self.m,
                self.batched.steel_gemm,
            )?;
        }
        Ok(())
    }

    /// One split point over `projs`, which all read `x` (`cols` halfs per
    /// row). `extras` encodes the class's Steel-only projections: in the
    /// running segment when the class does not split here, otherwise beside
    /// Steel's rows.
    fn point(
        &mut self,
        context: &mut gpu::MetalContext,
        class: Class,
        x: &gpu::MetalBuffer,
        cols: usize,
        projs: &[Proj<'_>],
        extras: impl FnOnce(&mut gpu::MetalContext, &gpu::PassEncoder) -> Result<(), RealForwardError>,
    ) -> Result<(), RealForwardError> {
        let Some(&(_, program)) = self.classes.iter().find(|(c, _)| *c == class) else {
            self.full(context, &self.pass, x, cols, projs)?;
            return extras(context, &self.pass);
        };
        let gpu_err = RealForwardError::Gpu;
        let state = self.state;
        let mats = projs
            .iter()
            .map(|p| resident_matrix(self.weights, self.index, &p.name, p.rows, cols))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(pi) = program {
            let prog = &state.programs[pi];
            for (i, w) in mats.iter().enumerate() {
                let plane = state.planes.get(prog.w[i]);
                ane::encode_ane_dequant_rows(
                    context,
                    &self.pass,
                    w,
                    prog.mas[i],
                    (&plane.buffer, 0),
                    plane.plane.stride(),
                )
                .map_err(gpu_err)?;
            }
            let a = state.planes.get(prog.a);
            ane::encode_ane_scale_rows(
                context,
                &self.pass,
                (x, 0),
                cols,
                self.m,
                (&a.buffer, 0),
                a.plane.stride(),
                &state.inv,
            )
            .map_err(gpu_err)?;
        }
        let steel = context.begin_pass_labeled("qwen split steel rows");
        match program {
            Some(pi) => {
                let prog = &state.programs[pi];
                for (i, (w, p)) in mats.iter().zip(projs).enumerate() {
                    ane::encode_dequant_int4_gemm_steel_rows(
                        context,
                        &steel,
                        w,
                        prog.mas[i],
                        (x, 0),
                        (p.out, 0),
                        p.rows,
                        self.m,
                    )
                    .map_err(gpu_err)?;
                }
            }
            None => self.full(context, &steel, x, cols, projs)?,
        }
        extras(context, &steel)?;
        let next = ane::begin_pass_after(
            context,
            "qwen split segment",
            &state.event,
            self.base + self.points.len() as u64 + 1,
        );
        if let Some(pi) = program {
            let prog = &state.programs[pi];
            for (i, p) in projs.iter().enumerate() {
                let o = state.planes.get(prog.o[i]);
                ane::encode_ane_convert_out(
                    context,
                    &next,
                    (&o.buffer, 0),
                    prog.mas[i],
                    self.m,
                    o.plane.stride(),
                    &state.inv,
                    (p.out, 0),
                    p.rows,
                    &self.nonfinite,
                )
                .map_err(gpu_err)?;
            }
        }
        self.segs.push(std::mem::replace(&mut self.pass, next));
        self.rows.push(steel);
        self.points.push((class, program));
        Ok(())
    }
}

/// Runs one dense micro-batch of `m` rows with the split. `first` already
/// holds the embedding lookups; `head` encodes (or skips) the LM head into
/// the last segment. Returns once every buffer has completed.
#[allow(clippy::too_many_arguments)]
fn run_micro_batch(
    context: &mut gpu::MetalContext,
    first: gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    kv: &mut gpu::KvCacheManager,
    qwen: &RealQwenState,
    arch: &ArchConfig,
    resid_capture: Option<&crate::resid_capture::ResidCapture>,
    steering: Option<&crate::steering::SteeringState>,
    mode: SplitMode,
    use_silu: bool,
    start_position: usize,
    m: usize,
    head: impl FnOnce(&mut gpu::MetalContext, &gpu::PassEncoder) -> Result<(), RealForwardError>,
) -> Result<SplitOutcome, RealForwardError> {
    let dims = Dims::new(arch, qwen.shape);
    let (hidden, inter) = (dims.hidden, dims.inter);
    let layers = arch.num_layers as usize;
    let unsupported = |what: &str| RealForwardError::Unsupported(format!("ane prefill: {what}"));
    let state = qwen
        .ane_split
        .as_ref()
        .ok_or_else(|| unsupported("state missing"))?;
    let batched = qwen.batched_prefill();
    let tokens = slot_for(&state.slots, m).ok_or_else(|| unsupported("no program slot"))?;
    let mut classes = Vec::new();
    for e in state.entries.iter().filter(|e| e.tokens == tokens) {
        let program = match mode {
            SplitMode::Break => None,
            SplitMode::Ane => Some(
                state
                    .programs
                    .iter()
                    .position(|p| p.class == e.class && p.tokens == tokens)
                    .ok_or_else(|| unsupported("program not loaded"))?,
            ),
        };
        classes.push((e.class, program));
    }
    // Every buffer is created before any is committed; past the queue's
    // capacity, creating one more blocks forever.
    let n_points: usize = (0..layers)
        .map(|l| {
            let linear = arch.layer_is_linear(l);
            classes.iter().filter(|(c, _)| c.in_layer(linear)).count()
        })
        .sum();
    if 2 * n_points + 1 > ane::MAX_COMMAND_BUFFERS {
        return Err(unsupported(&format!(
            "{} command buffers in flight, the queue allows {}",
            2 * n_points + 1,
            ane::MAX_COMMAND_BUFFERS
        )));
    }
    let kv: &gpu::KvCacheManager = kv;

    let t_encode = Instant::now();
    let mut sp = Split {
        weights,
        index,
        batched,
        state,
        classes,
        nonfinite: context.new_buffer_with_data(&[0u32]),
        m,
        base: state.event.value(),
        pass: first,
        segs: Vec::with_capacity(n_points + 1),
        rows: Vec::with_capacity(n_points),
        points: Vec::with_capacity(n_points),
    };
    let none = |_: &mut gpu::MetalContext, _: &gpu::PassEncoder| Ok(());
    for layer in 0..layers {
        let input_norm = norm_view(
            weights,
            index,
            &layer_tensor(layer, "input_layernorm.weight"),
            hidden,
        )?;
        let post_attn_norm = norm_view(
            weights,
            index,
            &layer_tensor(layer, "post_attention_layernorm.weight"),
            hidden,
        )?;
        encode_qwen_dense_input_norm_batched(
            context, &sp.pass, scratch, batched, input_norm, hidden, m,
        )?;
        if arch.layer_is_linear(layer) {
            let name = |s: &str| layer_tensor(layer, &format!("linear_attn.{s}"));
            sp.point(
                context,
                Class::Qkvz,
                &batched.normed,
                hidden,
                &[
                    Proj {
                        name: name("in_proj_qkv.weight"),
                        rows: dims.qkv_dim,
                        out: &batched.gdn_qkv_raw,
                    },
                    Proj {
                        name: name("in_proj_z.weight"),
                        rows: dims.value_dim,
                        out: &batched.gdn_z,
                    },
                ],
                |context, pass| {
                    encode_linear_in_proj_batched(
                        context, pass, weights, index, arch, qwen, batched, layer, m, false,
                    )
                },
            )?;
            // No tape slot, as in the default arm.
            encode_linear_core_batched(
                context, &sp.pass, weights, index, qwen, batched, layer, m, None,
            )?;
            sp.point(
                context,
                Class::O,
                &batched.gdn_out,
                dims.value_dim,
                &[Proj {
                    name: name("out_proj.weight"),
                    rows: hidden,
                    out: &batched.o,
                }],
                none,
            )?;
        } else {
            let slots = full_attention_kv_slots(arch, kv, layer, start_position)?;
            let name = |s: &str| layer_tensor(layer, &format!("self_attn.{s}"));
            sp.point(
                context,
                Class::Qg,
                &batched.normed,
                hidden,
                &[Proj {
                    name: name("q_proj.weight"),
                    rows: 2 * dims.q_dim,
                    out: &batched.q_packed,
                }],
                |context, pass| {
                    encode_full_attention_in_proj_batched(
                        context,
                        pass,
                        weights,
                        index,
                        arch,
                        batched,
                        slots,
                        TRUNK_PREFIX,
                        layer,
                        m,
                        false,
                    )
                },
            )?;
            encode_full_attention_core_batched(
                context,
                &sp.pass,
                weights,
                index,
                arch,
                qwen,
                scratch,
                batched,
                slots,
                TRUNK_PREFIX,
                layer,
                QkNormConvention::Plain,
                start_position,
                m,
            )?;
            sp.point(
                context,
                Class::O,
                &batched.attn_out,
                dims.q_dim,
                &[Proj {
                    name: name("o_proj.weight"),
                    rows: hidden,
                    out: &batched.o,
                }],
                none,
            )?;
        }
        encode_qwen_dense_attn_residual_batched(
            context,
            &sp.pass,
            scratch,
            batched,
            post_attn_norm,
            hidden,
            m,
        )?;
        let mlp = |s: &str| layer_tensor(layer, &format!("mlp.{s}"));
        sp.point(
            context,
            Class::GateUp,
            &batched.moe_x,
            hidden,
            &[
                Proj {
                    name: mlp("gate_proj.weight"),
                    rows: inter,
                    out: &batched.ffn_gate,
                },
                Proj {
                    name: mlp("up_proj.weight"),
                    rows: inter,
                    out: &batched.ffn_up,
                },
            ],
            none,
        )?;
        encode_dense_ffn_act_batched(context, &sp.pass, batched, inter, use_silu, m)?;
        sp.point(
            context,
            Class::Down,
            &batched.ffn_act,
            inter,
            &[Proj {
                name: mlp("down_proj.weight"),
                rows: hidden,
                out: &batched.h2,
            }],
            none,
        )?;
        encode_dense_ffn_residual_batched(context, &sp.pass, scratch, batched, hidden, m)?;
        encode_qwen_dense_layer_batched_post_ffn(
            context,
            &sp.pass,
            scratch,
            resid_capture,
            steering,
            layer,
            hidden,
            m,
        )?;
    }
    let Split {
        pass,
        mut segs,
        rows,
        points,
        base,
        nonfinite,
        ..
    } = sp;
    head(context, &pass)?;
    segs.push(pass);
    let encode_s = t_encode.elapsed().as_secs_f64();
    let n = points.len();

    // The recurrent state as of before this micro-batch, so a failure can
    // redo it on Steel. A blit committed ahead of seg(0) on the same queue,
    // so it reads the state before any of this micro-batch's passes write it.
    let snapshot = (mode == SplitMode::Ane).then(|| qwen.gdn.snapshot(context.queue()));

    // Commit in queue order: seg(0), rows(0), seg(1), rows(1), ..., seg(n).
    let t_commit = ane::host_seconds();
    let mut seg_done = Vec::with_capacity(n + 1);
    let mut rows_done = Vec::with_capacity(n);
    let mut rows_iter = rows.into_iter();
    for seg in segs {
        seg_done.push(seg.commit());
        if let Some(r) = rows_iter.next() {
            rows_done.push(r.commit());
        }
    }

    // The host half. A failed prediction still signals every value so the
    // GPU drains before the micro-batch is redone.
    let fault = if mode == SplitMode::Ane {
        take_fault_point()
    } else {
        None
    };
    let mut ready = vec![0.0f64; n];
    let mut predicted = vec![0.0f64; n];
    let mut signaled = vec![0.0f64; n];
    let mut failure: Option<String> = None;
    for (p, &(_, program)) in points.iter().enumerate() {
        seg_done[p].wait_ref();
        ready[p] = ane::host_seconds();
        if let (Some(pi), None) = (program, failure.as_ref()) {
            let result = if fault == Some(p) {
                Err("an injected fault (TURBOSPARK_ANE_PREFILL_FAULT)".to_string())
            } else {
                predict(&state.programs[pi], &state.planes)
            };
            if let Err(e) = result {
                failure = Some(format!("prediction {p} failed: {e}"));
            }
        }
        predicted[p] = ane::host_seconds();
        state.event.signal(base + p as u64 + 1);
        signaled[p] = ane::host_seconds();
    }
    seg_done[n].wait_ref();
    let wall = ane::host_seconds() - t_commit;
    if failure.is_none() && gpu::read_buffer_bytes(&nonfinite, 0, 4) != [0u8; 4] {
        failure = Some("non-finite ANE rows".into());
    }
    if let (Some(reason), Some(snapshot)) = (failure, snapshot) {
        return Ok(SplitOutcome::Redo { snapshot, reason });
    }

    let seg_t: Vec<(f64, f64)> = seg_done.iter().map(|c| c.wait_gpu_start_end()).collect();
    let rows_t: Vec<(f64, f64)> = rows_done.iter().map(|c| c.wait_gpu_start_end()).collect();
    let mut gpu_busy = 0.0;
    for (i, &(s, e)) in seg_t.iter().enumerate() {
        // seg(i) for i >= 1 waits on the signal sent after point i - 1.
        let from = if i == 0 { s } else { s.max(signaled[i - 1]) };
        gpu_busy += (e - from).max(0.0);
    }
    for &(rs, re) in &rows_t {
        gpu_busy += re - rs;
    }
    if std::env::var_os("TURBOSPARK_ANE_PREFILL_STATS").is_some() {
        print_stats(
            mode,
            start_position,
            m,
            tokens,
            encode_s,
            wall,
            gpu_busy,
            &points,
            &seg_t,
            &rows_t,
            &ready,
            &predicted,
            &signaled,
        );
    }
    Ok(SplitOutcome::Done(SplitTimes { gpu_busy, wall }))
}

/// One stderr line of a split micro-batch's counters, in milliseconds:
/// totals, then per class the Steel rows buffers, the predictions, the GPU
/// waiting on the ANE and the ANE waiting on the GPU.
#[allow(clippy::too_many_arguments)]
fn print_stats(
    mode: SplitMode,
    start_position: usize,
    m: usize,
    tokens: usize,
    encode_s: f64,
    wall: f64,
    gpu_busy: f64,
    points: &[(Class, Option<usize>)],
    seg_t: &[(f64, f64)],
    rows_t: &[(f64, f64)],
    ready: &[f64],
    predicted: &[f64],
    signaled: &[f64],
) {
    let (mut steel_rows, mut ane_s, mut wake, mut gpu_waits_ane, mut ane_waits_gpu, mut resume) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut per_class = [[0.0f64; 5]; 5];
    for (p, (class, _)) in points.iter().enumerate() {
        let (rs, re) = rows_t[p];
        steel_rows += re - rs;
        let pred = predicted[p] - ready[p];
        ane_s += pred;
        // GPU end of seg(p) to the host seeing it.
        wake += ready[p] - seg_t[p].1;
        // The Steel rows finished before the host's signal: GPU idle.
        let gw = (signaled[p] - re).max(0.0);
        gpu_waits_ane += gw;
        // The prediction finished before the Steel rows: ANE idle.
        let aw = (re - predicted[p]).max(0.0);
        ane_waits_gpu += aw;
        // Signal (or Steel end, if later) to the next segment's first work.
        resume += seg_t[p + 1].0 - signaled[p].max(re);
        let c = &mut per_class[Class::ALL.iter().position(|x| x == class).expect("listed")];
        c[0] += re - rs;
        c[1] += pred;
        c[2] += gw;
        c[3] += aw;
        c[4] += 1.0;
    }
    let mut by_class = String::new();
    for (class, c) in Class::ALL.iter().zip(per_class) {
        if c[4] > 0.0 {
            by_class.push_str(&format!(
                " {0}_n={1} {0}_rows_ms={2:.3} {0}_ane_ms={3:.3} {0}_gwait_ms={4:.3} \
                 {0}_await_ms={5:.3}",
                class.name(),
                c[4] as usize,
                c[0] * 1e3,
                c[1] * 1e3,
                c[2] * 1e3,
                c[3] * 1e3
            ));
        }
    }
    eprintln!(
        "[ane-prefill] mode={mode:?} start={start_position} m={m} slot={tokens} points={} \
         encode_ms={:.3} wall_ms={:.3} gpu_busy_ms={:.3} steel_rows_ms={:.3} ane_ms={:.3} \
         wake_ms={:.3} gpu_waits_ane_ms={:.3} ane_waits_gpu_ms={:.3} resume_ms={:.3}{by_class}",
        points.len(),
        encode_s * 1e3,
        wall * 1e3,
        gpu_busy * 1e3,
        steel_rows * 1e3,
        ane_s * 1e3,
        wake * 1e3,
        gpu_waits_ane * 1e3,
        ane_waits_gpu * 1e3,
        resume * 1e3,
    );
}

#[cfg(test)]
mod manifest_tests {
    use super::*;

    // Qwen3.8-27B.
    const DIMS: Dims = Dims {
        hidden: 5120,
        inter: 17408,
        q_dim: 6144,
        qkv_dim: 10240,
        value_dim: 6144,
    };

    fn manifest(programs: &str) -> String {
        format!(r#"{{"format": 1, "programs": [{programs}]}}"#)
    }

    const GATEUP_2048: &str = r#"{"class": "gateup", "k": 5120, "tokens": 2048,
        "matrices": [{"rows": 17408, "ane_rows": 8192, "plane_rows": 8192},
                     {"rows": 17408, "ane_rows": 8192, "plane_rows": 8192}],
        "package": "split_k5120_n2048_m8192w8192_m8192w8192.mlpackage"}"#;

    #[test]
    fn a_matching_manifest_parses() {
        let down = r#"{"class": "down", "k": 17408, "tokens": 1024,
            "matrices": [{"rows": 5120, "ane_rows": 1920, "plane_rows": 2272}],
            "package": "split_k17408_n1024_m1920w2272.mlpackage"}"#;
        let entries = parse_manifest(&manifest(&format!("{GATEUP_2048}, {down}")), DIMS).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].class, Class::GateUp);
        assert_eq!(entries[0].mats, vec![(8192, 8192), (8192, 8192)]);
        assert_eq!(entries[1].mats, vec![(1920, 2272)]);
    }

    #[test]
    fn a_manifest_for_another_shape_is_refused() {
        // A 4096-wide model's gate/up.
        let other = GATEUP_2048.replace("\"k\": 5120", "\"k\": 4096");
        let err = parse_manifest(&manifest(&other), DIMS).unwrap_err();
        assert!(err.contains("the install needs K 5120"), "{err}");
        let rows = GATEUP_2048.replacen("\"rows\": 17408", "\"rows\": 13824", 1);
        assert!(parse_manifest(&manifest(&rows), DIMS).is_err());
    }

    #[test]
    fn ane_rows_steel_cannot_take_are_refused() {
        for bad in ["8200", "0", "17408"] {
            let m = GATEUP_2048.replacen("\"ane_rows\": 8192", &format!("\"ane_rows\": {bad}"), 1);
            assert!(
                parse_manifest(&manifest(&m), DIMS).is_err(),
                "ane_rows {bad}"
            );
        }
        let small_plane = GATEUP_2048.replacen("\"plane_rows\": 8192", "\"plane_rows\": 4096", 1);
        assert!(parse_manifest(&manifest(&small_plane), DIMS).is_err());
    }

    #[test]
    fn other_manifest_errors_are_refused() {
        assert!(parse_manifest(&manifest(""), DIMS).is_err());
        assert!(parse_manifest("{", DIMS).is_err());
        let v2 = manifest(GATEUP_2048).replace("\"format\": 1", "\"format\": 2");
        assert!(parse_manifest(&v2, DIMS).is_err());
        let twice = manifest(&format!("{GATEUP_2048}, {GATEUP_2048}"));
        assert!(parse_manifest(&twice, DIMS).unwrap_err().contains("twice"));
        let wide = GATEUP_2048.replace("\"tokens\": 2048", "\"tokens\": 4096");
        assert!(parse_manifest(&manifest(&wide), DIMS).is_err());
        let path = GATEUP_2048.replace("\"package\": \"", "\"package\": \"../");
        assert!(parse_manifest(&manifest(&path), DIMS).is_err());
        let class = GATEUP_2048.replace("gateup", "lm_head");
        assert!(parse_manifest(&manifest(&class), DIMS).is_err());
    }

    #[test]
    fn micro_batches_take_the_smallest_count_that_fits() {
        let slots = [1024, 2048];
        assert_eq!(slot_for(&slots, 511), None);
        assert_eq!(slot_for(&slots, 512), Some(1024));
        assert_eq!(slot_for(&slots, 892), Some(1024));
        assert_eq!(slot_for(&slots, 1024), Some(1024));
        assert_eq!(slot_for(&slots, 1025), Some(2048));
        assert_eq!(slot_for(&slots, 2048), Some(2048));
        assert_eq!(slot_for(&slots, 2049), None);
        assert_eq!(slot_for(&[], 1024), None);
    }

    #[test]
    fn env_values_pick_the_mode() {
        for (v, mode) in [
            ("auto", AnePrefillMode::Auto),
            ("1", AnePrefillMode::Split),
            ("on", AnePrefillMode::Split),
            ("break", AnePrefillMode::Break),
            ("", AnePrefillMode::Off),
            ("0", AnePrefillMode::Off),
            ("yes", AnePrefillMode::Off),
        ] {
            assert_eq!(AnePrefillMode::from_env_value(v), mode, "{v:?}");
        }
    }

    #[test]
    fn parse_recognises_modes() {
        assert_eq!(AnePrefillMode::parse("auto"), Some(AnePrefillMode::Auto));
        assert_eq!(AnePrefillMode::parse("split"), Some(AnePrefillMode::Split));
        assert_eq!(AnePrefillMode::parse("on"), Some(AnePrefillMode::Split));
        assert_eq!(AnePrefillMode::parse("1"), Some(AnePrefillMode::Split));
        assert_eq!(AnePrefillMode::parse("break"), Some(AnePrefillMode::Break));
        assert_eq!(AnePrefillMode::parse("off"), Some(AnePrefillMode::Off));
        assert_eq!(AnePrefillMode::parse("0"), Some(AnePrefillMode::Off));
        assert_eq!(AnePrefillMode::parse("invalid"), None);
    }

    #[test]
    fn chip_detection_identifies_m4_and_newer() {
        assert!(chip_is_m4_or_newer("Apple M4"));
        assert!(chip_is_m4_or_newer("Apple M4 Pro"));
        assert!(chip_is_m4_or_newer("Apple M4 Max"));
        assert!(chip_is_m4_or_newer("Apple M4 Ultra"));
        assert!(chip_is_m4_or_newer("Apple M5"));
        assert!(chip_is_m4_or_newer("Apple M5 Max"));
        assert!(chip_is_m4_or_newer("Apple A18 Pro"));

        assert!(!chip_is_m4_or_newer("Apple M1"));
        assert!(!chip_is_m4_or_newer("Apple M1 Pro"));
        assert!(!chip_is_m4_or_newer("Apple M1 Max"));
        assert!(!chip_is_m4_or_newer("Apple M2"));
        assert!(!chip_is_m4_or_newer("Apple M2 Ultra"));
        assert!(!chip_is_m4_or_newer("Apple M3"));
        assert!(!chip_is_m4_or_newer("Apple M3 Max"));
        assert!(!chip_is_m4_or_newer("Apple A17 Pro"));
        assert!(!chip_is_m4_or_newer("Intel(R) Core(TM) i9-9980HK"));
        assert!(!chip_is_m4_or_newer(""));
    }
}

#[cfg(test)]
mod tests {
    //! Each manifest program's split on synthetic INT4 weights at its real
    //! shape, against one-shot Steel, plus the non-finite flag. Needs the
    //! Neural Engine and a programs directory
    //! (`crates/coreml/tools/gen_ane_prefill_programs.py`):
    //!
    //!   TURBOSPARK_ANE_PREFILL_DIR=~/models/ane-prefill/qwen38 cargo test \
    //!     -p turbospark-runtime --features ane-prefill --release \
    //!     ane_split -- --ignored --nocapture
    use super::*;
    use half::{bf16, f16};

    /// The synthetic recipe of `qwen38_int4_prefill_gemm_probe.rs`, with
    /// per-group scales that vary so a group-index bug cannot hide.
    fn blob(rows: usize, k: usize, seed: u64) -> (Vec<u8>, u64, u64) {
        let mut state = seed | 1;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let groups = rows * k / 64;
        let mut out: Vec<u8> = (0..rows * k / 2).map(|_| next() as u8).collect();
        let scales = out.len() as u64;
        for g in 0..groups {
            let s = 0.004 + 0.002 * (g % 7) as f32;
            out.extend_from_slice(&bf16::from_f32(s).to_bits().to_le_bytes());
        }
        let biases = out.len() as u64;
        for g in 0..groups {
            let b = -0.03 - 0.005 * (g % 5) as f32;
            out.extend_from_slice(&bf16::from_f32(b).to_bits().to_le_bytes());
        }
        (out, scales, biases)
    }

    /// Runs `prog`'s split of `ws` over `x` (m rows) and returns each
    /// matrix's output and whether the non-finite flag rose.
    fn split(
        context: &mut gpu::MetalContext,
        state: &AneSplitState,
        prog: &Program,
        ws: &[gpu::Int4ResidentMatrix<'_>],
        x: &gpu::MetalBuffer,
        k: usize,
        m: usize,
    ) -> (Vec<Vec<f16>>, bool) {
        let ys: Vec<_> = ws
            .iter()
            .map(|w| context.new_output_buffer((m * w.rows * 2) as u64))
            .collect();
        let flag = context.new_buffer_with_data(&[0u32]);
        let pass = context.begin_pass();
        for (i, w) in ws.iter().enumerate() {
            let p = state.planes.get(prog.w[i]);
            ane::encode_ane_dequant_rows(
                context,
                &pass,
                w,
                prog.mas[i],
                (&p.buffer, 0),
                p.plane.stride(),
            )
            .unwrap();
        }
        let ap = state.planes.get(prog.a);
        ane::encode_ane_scale_rows(
            context,
            &pass,
            (x, 0),
            k,
            m,
            (&ap.buffer, 0),
            ap.plane.stride(),
            &state.inv,
        )
        .unwrap();
        for (i, (w, y)) in ws.iter().zip(&ys).enumerate() {
            ane::encode_dequant_int4_gemm_steel_rows(
                context,
                &pass,
                w,
                prog.mas[i],
                (x, 0),
                (y, 0),
                w.rows,
                m,
            )
            .unwrap();
        }
        pass.commit_and_wait();
        predict(prog, &state.planes).expect("predict");
        let pass = context.begin_pass();
        for (i, (w, y)) in ws.iter().zip(&ys).enumerate() {
            let op = state.planes.get(prog.o[i]);
            ane::encode_ane_convert_out(
                context,
                &pass,
                (&op.buffer, 0),
                prog.mas[i],
                m,
                op.plane.stride(),
                &state.inv,
                (y, 0),
                w.rows,
                &flag,
            )
            .unwrap();
        }
        pass.commit_and_wait();
        let out = ws
            .iter()
            .zip(&ys)
            .map(|(w, y)| gpu::read_buffer_f16(y, 0, m * w.rows))
            .collect();
        (out, gpu::read_buffer_bytes(&flag, 0, 4) != [0u8; 4])
    }

    #[test]
    #[ignore = "needs the Neural Engine and TURBOSPARK_ANE_PREFILL_DIR"]
    fn ane_split_matches_one_shot_steel() {
        let Some(dir) = std::env::var_os("TURBOSPARK_ANE_PREFILL_DIR") else {
            eprintln!("TURBOSPARK_ANE_PREFILL_DIR is not set; skipping");
            return;
        };
        let dims = Dims {
            hidden: 5120,
            inter: 17408,
            q_dim: 6144,
            qkv_dim: 10240,
            value_dim: 6144,
        };
        let mut context = gpu::MetalContext::new().expect("Metal");
        let mut state = AneSplitState::open(&context, Path::new(&dir), dims);
        assert!(state.off.is_none(), "{:?}", state.off);
        state.load_programs(&context).expect("programs");
        // The smallest token count, 24 rows short so padding is exercised.
        let n = state.slots[0];
        let m = n - 24;
        let mut worst_all = 0.0f64;
        for (pi, prog) in state
            .programs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.tokens == n)
        {
            let (k, rows) = dims.class(prog.class);
            // Rows of very different magnitudes, so the per-token scale
            // matters: without it the 1e-3 rows sit in F16 subnormals on
            // the ANE.
            let mags = [1.0f32, 1e-3, 30.0, 0.05, 4.0];
            let x: Vec<u16> = (0..m * k)
                .map(|i| {
                    let (t, kk) = (i / k, i % k);
                    let v = (((t * 7 + kk * 13) % 29) as f32 - 14.0) / 64.0 * mags[t % mags.len()];
                    f16::from_f32(v).to_bits()
                })
                .collect();
            let xbuf = context.new_buffer_with_data(&x);
            let wbufs: Vec<_> = rows
                .iter()
                .enumerate()
                .map(|(i, &r)| {
                    let (data, so, bo) = blob(r, k, 0xC0FFEE + (pi * 2 + i) as u64);
                    (context.new_buffer_with_data(&data), so, bo)
                })
                .collect();
            let ws: Vec<gpu::Int4ResidentMatrix<'_>> = wbufs
                .iter()
                .zip(&rows)
                .map(|((buf, so, bo), &r)| gpu::Int4ResidentMatrix {
                    buffer: buf,
                    weights_offset: 0,
                    scales_offset: *so,
                    biases_offset: *bo,
                    rows: r,
                    cols: k,
                })
                .collect();
            let refs: Vec<Vec<f16>> = ws
                .iter()
                .map(|w| {
                    let y = context.new_output_buffer((m * w.rows * 2) as u64);
                    let pass = context.begin_pass();
                    gpu::encode_dequant_int4_gemm_steel(
                        &mut context,
                        &pass,
                        w,
                        (&xbuf, 0),
                        (&y, 0),
                        m,
                        gpu::SteelTile::M32K32N32,
                    )
                    .unwrap();
                    pass.commit_and_wait();
                    gpu::read_buffer_f16(&y, 0, m * w.rows)
                })
                .collect();
            let (got, flagged) = split(&mut context, &state, prog, &ws, &xbuf, k, m);
            assert!(!flagged, "{:?}: non-finite flag on finite data", prog.class);
            for (i, &r) in rows.iter().enumerate() {
                let ma = prog.mas[i];
                let (want, got) = (&refs[i], &got[i]);
                // Steel's rows land at their offset, bit for bit.
                let steel_diff = (0..m)
                    .flat_map(|t| (ma..r).map(move |row| t * r + row))
                    .filter(|&j| want[j].to_bits() != got[j].to_bits())
                    .count();
                // The ANE rows: worst per-token max |d| / max |ref| (Phase
                // 0's metric).
                let mut worst = 0.0f64;
                for t in 0..m {
                    let (mut mx, mut md) = (0.0f64, 0.0f64);
                    for row in 0..ma {
                        let (a, b) = (want[t * r + row].to_f64(), got[t * r + row].to_f64());
                        mx = mx.max(a.abs());
                        md = md.max((a - b).abs());
                    }
                    worst = worst.max(if mx > 0.0 { md / mx } else { md });
                }
                eprintln!(
                    "{} matrix {i} ({r} x {k}, ANE rows {ma}): Steel rows differing {steel_diff}, \
                     ANE rows worst normalized error {worst:.5}",
                    prog.class.name()
                );
                assert_eq!(steel_diff, 0, "{:?}/{i}: Steel rows differ", prog.class);
                worst_all = worst_all.max(worst);
            }
            // Activations near the F16 limit overflow the output: the flag
            // must rise (Steel's output overflows too).
            let huge = context.new_buffer_with_data(&vec![f16::from_f32(60000.0).to_bits(); m * k]);
            let (_, flagged) = split(&mut context, &state, prog, &ws, &huge, k, m);
            assert!(
                flagged,
                "{:?}: overflowing output left the flag down",
                prog.class
            );
        }
        assert!(worst_all < 5e-3, "ANE rows off by {worst_all}");
    }
}
