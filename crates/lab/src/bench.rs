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
    crate::say_part!(
        "  {:<16}",
        format!("{} at {} =", if use_c { "c(M)" } else { "ms" }, r.x.name)
    );
    for x in &xs {
        crate::say_part!("{:>9}", x);
    }
    crate::say!("   fit a + b*x (ms)");
    for a in &r.arms {
        if let Some(why) = &a.unsupported {
            crate::say!("  {:<16}{why}", a.label);
            continue;
        }
        crate::say_part!("  {:<16}", a.label);
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
    for n in &r.notes {
        crate::say!("  note: {n}");
    }
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
