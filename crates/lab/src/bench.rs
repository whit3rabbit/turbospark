//! The one result shape every kernel bench emits.
//!
//! A bench used to write its own JSON (`int4_matmul` keyed points by `m`,
//! `attn_decode` nested kernels under each context), so plotting a new
//! kernel meant new reader code. Now a bench fills a [`BenchResult`]: what
//! was measured (`subject`), along which axis (`x`), and one curve per arm.
//! The ledger (`kernel_curve`), `--json`, the export and the HTML report all
//! read this shape, so a new bench is visible everywhere with no other code.
//!
//! Portable plain data; the benches that fill it are macOS-only.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::fit::Affine;

/// The layout `kernel::support::int4_blob` synthesizes: MLX affine INT4,
/// group 64, BF16 scales and biases, the resident order. Version-0 int4
/// records used it too.
pub const INT4_AFFINE_LAYOUT: &str = "mlx_affine_int4_g64_bf16";

/// The swept variable of a bench.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Axis {
    /// `m` (rows), `context` (KV positions), `tokens`, `config` (an index
    /// into `subject.configs`) or `n` (probe count).
    pub name: String,
    /// `rows`, `positions`, `tokens`, `index` or `count`.
    pub unit: String,
    /// Plot hint: `log2` or `linear`.
    pub scale: String,
}

impl Axis {
    pub fn new(name: &str, unit: &str, scale: &str) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            scale: scale.into(),
        }
    }
}

/// One measured point of one arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    /// Median GPU milliseconds per sample.
    pub ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gb_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tflops: Option<f64>,
    /// Cost of x rows in one pass over x one-row passes, where the bench has
    /// a one-row unit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c_of_m: Option<f64>,
    /// FNV-1a 64 of the output bytes, hex, under `--hash`. Equal hashes on
    /// two builds mean the arm's arithmetic did not change at this point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_fnv: Option<String>,
    /// FNV-1a 64 of the first output row alone, hex, under `--hash`. A row's
    /// bits that do not move as the batch around it grows is the property
    /// that lets decode and verify share one arithmetic (`row_invariant` in
    /// [`exactness_hints`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_fnv: Option<String>,
}

impl Point {
    pub fn new(x: f64, ms: f64) -> Self {
        Self {
            x,
            ms,
            gb_s: None,
            tflops: None,
            c_of_m: None,
            out_fnv: None,
            row_fnv: None,
        }
    }
}

/// One arm's curve. An arm the kernel cannot serve at this case carries
/// `unsupported` and no points, so a reader can tell "not measured" from
/// "missing".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmCurve {
    pub label: String,
    #[serde(default)]
    pub points: Vec<Point>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<Affine>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsupported: Option<String>,
}

impl ArmCurve {
    pub fn unsupported(label: &str, why: &str) -> Self {
        Self {
            label: label.into(),
            points: Vec::new(),
            fit: None,
            unsupported: Some(why.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchResult {
    /// The registered bench name (`tslab kernel --bench NAME`).
    pub bench: String,
    /// Short case label, unique within the bench for one install:
    /// `17408x5120`, `full_attn`, `sliding_attn`, ...
    pub case: String,
    /// What was measured: shape parameters and bench constants.
    pub subject: BTreeMap<String, Value>,
    pub x: Axis,
    /// The weight or cache layout the bench synthesized. A reader compares
    /// it with the install's own layout before trusting a curve.
    pub layout: String,
    /// Arms in the order the bench ran them.
    pub arms: Vec<ArmCurve>,
    /// Pairwise break-even points: the x above which `arm` beats `against`.
    #[serde(default)]
    pub break_even: Vec<BreakEven>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BreakEven {
    pub arm: String,
    pub against: String,
    /// `None`: never beats it above any x.
    pub above_x: Option<f64>,
}

impl BenchResult {
    pub fn arm(&self, label: &str) -> Option<&ArmCurve> {
        self.arms.iter().find(|a| a.label == label)
    }
}

/// The human table for one result: one row per arm, one column per x. Rows
/// show c(M) when every point has it (the int4 curves), milliseconds
/// otherwise. The ledger holds every field; this is only a reading aid.
pub fn print_result(r: &BenchResult) {
    let subject: Vec<String> = r
        .subject
        .iter()
        .filter(|(_, v)| !v.is_object() && !v.is_array())
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    crate::say!(
        "{} {}: {} [{}]",
        r.bench,
        r.case,
        subject.join(" "),
        r.layout
    );
    // A hash corpus has no timing: say how many points it hashed instead
    // of a table of zeros.
    let hashed_only = r
        .arms
        .iter()
        .flat_map(|a| &a.points)
        .all(|p| p.ms == 0.0 && p.out_fnv.is_some());
    if hashed_only && r.arms.iter().any(|a| !a.points.is_empty()) {
        for a in &r.arms {
            crate::say!("  {}: {} output hashes", a.label, a.points.len());
        }
        for n in &r.notes {
            crate::say!("  note: {n}");
        }
        return;
    }
    let xs: Vec<f64> = {
        let mut v: Vec<f64> = r
            .arms
            .iter()
            .flat_map(|a| a.points.iter().map(|p| p.x))
            .collect();
        v.sort_by(f64::total_cmp);
        v.dedup();
        v
    };
    let use_c = r
        .arms
        .iter()
        .filter(|a| !a.points.is_empty())
        .all(|a| a.points.iter().all(|p| p.c_of_m.is_some()));
    let w = r
        .arms
        .iter()
        .map(|a| a.label.len())
        .max()
        .unwrap_or(0)
        .max(14)
        + 2;
    crate::say_part!(
        "  {:<w$}",
        format!("{} at {} =", if use_c { "c(M)" } else { "ms" }, r.x.name)
    );
    for x in &xs {
        crate::say_part!("{:>9}", x);
    }
    crate::say!("   fit a + b*x (ms)");
    for a in &r.arms {
        if let Some(why) = &a.unsupported {
            crate::say!("  {:<w$}{why}", a.label);
            continue;
        }
        crate::say_part!("  {:<w$}", a.label);
        for x in &xs {
            match a.points.iter().find(|p| p.x == *x) {
                Some(p) => {
                    let v = if use_c {
                        p.c_of_m.unwrap_or(f64::NAN)
                    } else {
                        p.ms
                    };
                    crate::say_part!("{v:>9.3}")
                }
                None => crate::say_part!("{:>9}", "-"),
            }
        }
        if let Some(f) = &a.fit {
            crate::say_part!("   {:.4} + {:.4}x", f.a, f.b);
        }
        crate::say!();
    }
    for b in &r.break_even {
        crate::say!(
            "  {} beats {} above {} = {}",
            b.arm,
            b.against,
            r.x.name,
            b.above_x.map_or("never".to_string(), |m| format!("{m:.1}"))
        );
    }
    for h in exactness_hints(r) {
        if h.bit_exact_vs_first.is_none() && h.row_invariant.is_none() {
            continue;
        }
        let word = |v: Option<bool>, yes: &str, no: &str| match v {
            Some(true) => yes.to_string(),
            Some(false) => no.to_string(),
            None => "unhashed".to_string(),
        };
        crate::say!(
            "  exactness {}: vs {} {}; row {}",
            h.arm,
            h.against,
            word(h.bit_exact_vs_first, "bit-equal", "DIFFERS"),
            word(h.row_invariant, "invariant", "VARIES")
        );
    }
    for n in &r.notes {
        crate::say!("  note: {n}");
    }
}

/// Benches whose arms are different kernels rather than alternatives.
const DISTINCT_KERNEL_BENCHES: &[&str] = &["gdn_delta"];

/// What the hashes of one arm say about its arithmetic, read off a hashed
/// run. `None` means the run did not hash enough points to say.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArmHint {
    pub arm: String,
    /// The arm the others are compared with (the first arm that ran).
    pub against: String,
    /// Output bits equal to `against` at every x both hashed. False means the
    /// arm is a different arithmetic: a drift bound or a shared-arithmetic
    /// mode, never a `bit_exact` spec.
    pub bit_exact_vs_first: Option<bool>,
    /// The first row's bits equal at every x the arm hashed: the row does not
    /// depend on the rows beside it.
    pub row_invariant: Option<bool>,
}

/// Per-arm exactness hints from a result's hashes. The first arm is the
/// reference and carries only its own `row_invariant`. These are hints from
/// one case at one seed, never a gate: the parity tests are the gate.
pub fn exactness_hints(r: &BenchResult) -> Vec<ArmHint> {
    // The arms of these benches are different kernels (decode against fold),
    // not alternatives for one op, so "bit-equal to the first arm" is
    // meaningless. Their modes are compared across experiments with
    // `tslab hash-diff`; only row invariance is read here.
    let distinct_kernels = DISTINCT_KERNEL_BENCHES.contains(&r.bench.as_str());
    let first = r.arms.iter().find(|a| !a.points.is_empty());
    let mut out = Vec::new();
    for (i, a) in r.arms.iter().enumerate() {
        if a.points.is_empty() {
            continue;
        }
        let rows: Vec<&String> = a.points.iter().filter_map(|p| p.row_fnv.as_ref()).collect();
        let row_invariant = (rows.len() >= 2).then(|| rows.windows(2).all(|w| w[0] == w[1]));
        let (against, bit_exact_vs_first) = match first {
            Some(f) if !std::ptr::eq(f, a) && i > 0 && !distinct_kernels => {
                let mut pairs = 0usize;
                let mut equal = true;
                for p in &a.points {
                    let (Some(h), Some(q)) = (
                        p.out_fnv.as_ref(),
                        f.points
                            .iter()
                            .find(|q| q.x == p.x)
                            .and_then(|q| q.out_fnv.as_ref()),
                    ) else {
                        continue;
                    };
                    pairs += 1;
                    equal &= h == q;
                }
                (f.label.clone(), (pairs > 0).then_some(equal))
            }
            _ => (a.label.clone(), None),
        };
        out.push(ArmHint {
            arm: a.label.clone(),
            against,
            bit_exact_vs_first,
            row_invariant,
        });
    }
    out
}

/// FNV-1a 64 over raw bytes, as `music3_linear_bench` hashes its outputs.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

pub fn fnv_hex(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a64(bytes))
}

/// One hash mismatch between two runs of the same bench case.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HashChange {
    pub bench: String,
    pub case: String,
    pub arm: String,
    pub x: f64,
    pub before: String,
    pub after: String,
}

/// Compare the hashed points of two sets of bench results, matched on
/// (bench, case, arm, x). Points hashed on only one side are counted, not
/// reported as changes.
pub fn hash_diff(before: &[BenchResult], after: &[BenchResult]) -> (Vec<HashChange>, usize, usize) {
    let key = |r: &BenchResult, a: &ArmCurve, p: &Point| {
        (
            r.bench.clone(),
            r.case.clone(),
            a.label.clone(),
            p.x.to_bits(),
        )
    };
    let mut old = BTreeMap::new();
    for r in before {
        for a in &r.arms {
            for p in &a.points {
                if let Some(h) = &p.out_fnv {
                    old.insert(key(r, a, p), h.clone());
                }
            }
        }
    }
    let (mut changes, mut compared, mut unmatched) = (Vec::new(), 0, 0);
    for r in after {
        for a in &r.arms {
            for p in &a.points {
                let Some(h) = &p.out_fnv else { continue };
                match old.get(&key(r, a, p)) {
                    Some(prev) => {
                        compared += 1;
                        if prev != h {
                            changes.push(HashChange {
                                bench: r.bench.clone(),
                                case: r.case.clone(),
                                arm: a.label.clone(),
                                x: p.x,
                                before: prev.clone(),
                                after: h.clone(),
                            });
                        }
                    }
                    None => unmatched += 1,
                }
            }
        }
    }
    (changes, compared, unmatched)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hashed_arm(label: &str, pts: &[(f64, &str, &str)]) -> ArmCurve {
        ArmCurve {
            label: label.into(),
            points: pts
                .iter()
                .map(|(x, out, row)| {
                    let mut p = Point::new(*x, 1.0);
                    p.out_fnv = Some((*out).into());
                    p.row_fnv = Some((*row).into());
                    p
                })
                .collect(),
            fit: None,
            unsupported: None,
        }
    }

    fn with_arms(arms: Vec<ArmCurve>) -> BenchResult {
        BenchResult {
            bench: "int4_matmul".into(),
            case: "64x64".into(),
            subject: BTreeMap::new(),
            x: Axis::new("m", "rows", "log2"),
            layout: "mlx_affine_int4_g64".into(),
            arms,
            break_even: Vec::new(),
            notes: Vec::new(),
        }
    }

    #[test]
    fn hints_tell_a_bit_equal_arm_from_a_different_arithmetic() {
        let r = with_arms(vec![
            hashed_arm("gemv", &[(1.0, "a1", "r"), (8.0, "a8", "r")]),
            hashed_arm("same", &[(1.0, "a1", "r"), (8.0, "a8", "r")]),
            // Different bits, but the row does not move with the batch: the
            // shared-arithmetic signature.
            hashed_arm("tile", &[(1.0, "t1", "q"), (8.0, "t8", "q")]),
            // Same output at x = 1, then a row that changes with the batch.
            hashed_arm("loose", &[(1.0, "a1", "r"), (8.0, "x8", "z")]),
        ]);
        let h = exactness_hints(&r);
        let by = |name: &str| h.iter().find(|x| x.arm == name).unwrap();
        assert_eq!(by("gemv").bit_exact_vs_first, None, "the reference");
        assert_eq!(by("gemv").row_invariant, Some(true));
        assert_eq!(by("same").bit_exact_vs_first, Some(true));
        assert_eq!(by("tile").bit_exact_vs_first, Some(false));
        assert_eq!(by("tile").row_invariant, Some(true));
        assert_eq!(by("loose").bit_exact_vs_first, Some(false));
        assert_eq!(by("loose").row_invariant, Some(false));
    }

    #[test]
    fn arms_that_are_different_kernels_are_not_compared_with_each_other() {
        let mut r = with_arms(vec![
            hashed_arm("decode", &[(1.0, "d1", "r")]),
            hashed_arm("fold", &[(1.0, "f1", "q")]),
        ]);
        r.bench = "gdn_delta".into();
        let h = exactness_hints(&r);
        assert!(h.iter().all(|x| x.bit_exact_vs_first.is_none()), "{h:?}");
    }

    #[test]
    fn hints_say_nothing_about_unhashed_arms_or_unmatched_points() {
        let mut unhashed = hashed_arm("a", &[(1.0, "a1", "r")]);
        unhashed.points[0].out_fnv = None;
        unhashed.points[0].row_fnv = None;
        let r = with_arms(vec![
            unhashed,
            hashed_arm("b", &[(2.0, "b2", "r")]),
            ArmCurve::unsupported("c", "no tile"),
        ]);
        let h = exactness_hints(&r);
        assert_eq!(h.len(), 2, "an unsupported arm has no hint");
        let b = h.iter().find(|x| x.arm == "b").unwrap();
        assert_eq!(b.bit_exact_vs_first, None, "no shared x, no claim");
        assert_eq!(b.row_invariant, None, "one point is not an invariance");
    }

    fn result(hash: &str) -> BenchResult {
        let mut p = Point::new(8.0, 1.0);
        p.out_fnv = Some(hash.into());
        BenchResult {
            bench: "int4_matmul".into(),
            case: "64x64".into(),
            subject: BTreeMap::new(),
            x: Axis::new("m", "rows", "log2"),
            layout: "mlx_affine_int4_g64".into(),
            arms: vec![ArmCurve {
                label: "scalar_gemm".into(),
                points: vec![p],
                fit: None,
                unsupported: None,
            }],
            break_even: vec![],
            notes: vec![],
        }
    }

    #[test]
    fn fnv_matches_reference_vectors() {
        // Published FNV-1a 64 test vectors.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn hash_diff_flags_a_changed_point_only() {
        let (c, n, u) = hash_diff(&[result("aa")], &[result("aa")]);
        assert!(c.is_empty() && n == 1 && u == 0);
        let (c, n, _) = hash_diff(&[result("aa")], &[result("ab")]);
        assert_eq!((c.len(), n), (1, 1));
        assert_eq!(c[0].arm, "scalar_gemm");
        let mut other = result("aa");
        other.case = "128x64".into();
        let (c, n, u) = hash_diff(&[result("aa")], &[other]);
        assert!(c.is_empty() && n == 0 && u == 1);
    }

    #[test]
    fn round_trips_through_json() {
        let r = result("aa");
        let v = serde_json::to_value(&r).unwrap();
        assert!(v["arms"][0].get("unsupported").is_none());
        let back: BenchResult = serde_json::from_value(v).unwrap();
        assert_eq!(back, r);
    }
}
