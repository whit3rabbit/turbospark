//! Recipe: which target type each tensor gets.
//!
//! JSON, not TOML, so no new dependency is needed. Example:
//!
//! ```json
//! {
//!   "default": "q8_0",
//!   "fallback": "f16",
//!   "rules": [
//!     { "match": "blk.*.ffn_*_exps.weight", "type": "q4_0" },
//!     { "match": "*attn_output.weight", "type": "q5_0", "layers": [0, 3] },
//!     { "match": "token_embd.weight", "type": "keep" }
//!   ]
//! }
//! ```
//!
//! First matching rule wins. `*` matches any run of characters, dots included.
//! `layers` is an inclusive `[lo, hi]` range on the `blk.<N>.` prefix, and a
//! rule with `layers` never matches a tensor outside a `blk.<N>.` namespace.

use super::quantizers::Target;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// Copy the tensor bytes through unchanged.
    Keep,
    Type(Target),
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub pattern: String,
    pub layers: Option<(u32, u32)>,
    pub choice: Choice,
}

#[derive(Debug, Clone)]
pub struct Recipe {
    pub default: Choice,
    /// Used when a tensor's row length is not a whole number of target blocks.
    pub fallback: Target,
    pub rules: Vec<Rule>,
}

#[derive(Debug)]
pub struct RecipeError(pub String);

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "recipe: {}", self.0)
    }
}

impl std::error::Error for RecipeError {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecipe {
    default: String,
    fallback: Option<String>,
    #[serde(default)]
    rules: Vec<RawRule>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRule {
    #[serde(rename = "match")]
    pattern: String,
    #[serde(rename = "type")]
    ty: String,
    layers: Option<[u32; 2]>,
}

fn choice(s: &str) -> Result<Choice, RecipeError> {
    if s.eq_ignore_ascii_case("keep") {
        return Ok(Choice::Keep);
    }
    Target::parse(s).map(Choice::Type).ok_or_else(|| {
        RecipeError(format!(
            "unknown type {s:?} (keep, f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_k..q6_k, iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs)"
        ))
    })
}

impl Recipe {
    pub fn from_json(text: &str) -> Result<Recipe, RecipeError> {
        let raw: RawRecipe = serde_json::from_str(text).map_err(|e| RecipeError(e.to_string()))?;
        let fallback = match raw.fallback.as_deref() {
            None => Target::F16,
            Some(s) => Target::parse(s)
                .ok_or_else(|| RecipeError(format!("fallback {s:?} must be a concrete type")))?,
        };
        let mut rules = Vec::new();
        for r in raw.rules {
            if let Some([lo, hi]) = r.layers {
                if lo > hi {
                    return Err(RecipeError(format!(
                        "rule {:?}: layers [{lo}, {hi}] is an empty range",
                        r.pattern
                    )));
                }
            }
            rules.push(Rule {
                choice: choice(&r.ty)?,
                layers: r.layers.map(|[lo, hi]| (lo, hi)),
                pattern: r.pattern,
            });
        }
        Ok(Recipe {
            default: choice(&raw.default)?,
            fallback,
            rules,
        })
    }

    /// Serialize back to the JSON format [`Recipe::from_json`] reads.
    pub fn to_json(&self) -> String {
        let name = |c: &Choice| match c {
            Choice::Keep => "keep".to_string(),
            Choice::Type(t) => t.name().to_string(),
        };
        let rules: Vec<serde_json::Value> = self
            .rules
            .iter()
            .map(|r| {
                let mut o = serde_json::json!({ "match": r.pattern, "type": name(&r.choice) });
                if let Some((lo, hi)) = r.layers {
                    o["layers"] = serde_json::json!([lo, hi]);
                }
                o
            })
            .collect();
        serde_json::to_string_pretty(&serde_json::json!({
            "default": name(&self.default),
            "fallback": self.fallback.name(),
            "rules": rules,
        }))
        .expect("recipe serializes")
    }

    /// First matching rule, if any.
    pub fn rule_for(&self, name: &str) -> Option<&Rule> {
        let layer = layer_of(name);
        self.rules.iter().find(|r| {
            glob(&r.pattern, name)
                && match r.layers {
                    None => true,
                    Some((lo, hi)) => layer.is_some_and(|l| (lo..=hi).contains(&l)),
                }
        })
    }
}

/// `blk.<N>.rest` -> N.
fn layer_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// `*`-only glob (iterative, linear backtracking).
fn glob(pat: &str, s: &str) -> bool {
    let (p, s) = (pat.as_bytes(), s.as_bytes());
    let (mut pi, mut si) = (0, 0);
    let (mut star, mut mark) = (None, 0);
    while si < s.len() {
        if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            mark = si;
            pi += 1;
        } else if pi < p.len() && p[pi] == s[si] {
            pi += 1;
            si += 1;
        } else if let Some(st) = star {
            pi = st + 1;
            mark += 1;
            si = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == b'*')
}

#[cfg(test)]
mod glob_tests {
    use super::glob;

    #[test]
    fn matches() {
        assert!(glob(
            "blk.*.ffn_*_exps.weight",
            "blk.12.ffn_down_exps.weight"
        ));
        assert!(glob("*", ""));
        assert!(glob("token_embd.weight", "token_embd.weight"));
        assert!(!glob("token_embd.weight", "token_embd.weight2"));
        assert!(!glob("blk.*.attn_q.weight", "blk.1.attn_k.weight"));
    }
}
