//! Build a recipe from the tensor types of an existing GGUF.
//!
//! This is the faithful way to follow a published mix, such as one of
//! Unsloth's "UD" files. Their per-tensor choices are data driven and vary
//! layer by layer (for example one model mixes IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
//! IQ2_S and Q2_K within `ffn_down` alone), so no role-level rule set can
//! reproduce them. Reading the map out of the published file and applying it
//! to your own float copy of the same model does, without guessing at the
//! algorithm that produced it. It copies the *type assignment* only: the
//! values still come from your source and your imatrix.

use std::fs::File;
use std::path::Path;

use super::quantizers::Target;
use super::recipe::{Choice, Recipe, RecipeError, Rule};
use super::writer::read_header;
use crate::gguf_header::ggml_type_name;

/// One exact-name rule per tensor of `reference` with two or more dimensions.
/// 1-D tensors are left to the writer's copy rule. A reference tensor of a type
/// this crate cannot produce (for example Q8_1, which never appears in a model
/// file, or a type id from a fork this crate has not read) is an error naming it, because silently picking another type would
/// not be "the published mix".
pub fn recipe_from_gguf(reference: &Path) -> Result<Recipe, RecipeError> {
    let mut f =
        File::open(reference).map_err(|e| RecipeError(format!("{}: {e}", reference.display())))?;
    let len = f.metadata().map_err(|e| RecipeError(e.to_string()))?.len();
    let h = read_header(&mut f, len).map_err(|e| RecipeError(e.to_string()))?;

    let mut rules = Vec::new();
    let mut unsupported: Vec<String> = Vec::new();
    for (name, info) in &h.tensors {
        if info.dims.len() < 2 {
            continue;
        }
        match Target::from_ggml_id(info.ggml_type) {
            Some(t) => rules.push(Rule {
                pattern: name.clone(),
                layers: None,
                choice: Choice::Type(t),
            }),
            None => unsupported.push(format!(
                "{name} ({})",
                ggml_type_name(info.ggml_type).unwrap_or("unknown type")
            )),
        }
    }
    if !unsupported.is_empty() {
        let shown: Vec<_> = unsupported.iter().take(3).cloned().collect();
        return Err(RecipeError(format!(
            "{} reference tensors use types this crate cannot produce, e.g. {}",
            unsupported.len(),
            shown.join(", ")
        )));
    }
    Ok(Recipe {
        // A source tensor the reference does not name stays as it is, rather
        // than being quantized to a guessed type.
        default: Choice::Keep,
        fallback: Target::F16,
        rules,
    })
}
