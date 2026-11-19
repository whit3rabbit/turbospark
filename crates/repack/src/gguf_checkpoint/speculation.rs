//! Whether a GGUF file would speculate once installed, read off its header.
//!
//! The probe and the Hub check call this before a download so a reader can
//! tell an MTP-capable file from one that only decodes. It answers with the
//! runtime gate's own rules (`MtpState::speculation_blocker`): the verify
//! list and the batched dtype set are `model_io::mtp_verify`'s, shared by
//! both, because two copies drift the first time a kernel lands.
//!
//! Only dense `qwen35` has a GGUF head intake (`plan::classify`), so every
//! other family answers `None` ("this file has no speculation path to
//! predict"), never a guess.

use std::collections::BTreeMap;

use model_io::{ArchConfig, ModelFamily, MTP_BATCHED_GEMM_DTYPES, MTP_VERIFY_PROJECTIONS};

use super::plan::{classify, head_block_index};
use super::transcode::gdn_out_proj_needs_q8_0_requant;
use super::types::dtype_tag_for_ggml_type;
use crate::gguf_header::{ggml_type_name, GgufHeader};
use crate::gguf_names::{map_gguf_name_with_head, GgufMapping};

/// What a GGUF file's install would do about speculative decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufSpeculation {
    /// The head is ingested and every verify matrix has a batched kernel.
    Available,
    /// The file carries no multi-token-prediction head.
    NoHead,
    /// The file carries head tensors the install skips (an incomplete or
    /// unrecognized head), so the install has none.
    HeadIgnored { found: usize },
    /// The head is ingested, but this verify matrix would be stored at a
    /// dtype the batched verify has no kernel for.
    UnbatchedMatrix { tensor: String, ggml_type: u32 },
}

impl GgufSpeculation {
    pub fn is_available(&self) -> bool {
        matches!(self, GgufSpeculation::Available)
    }

    /// Why the install would not speculate, or `None` when it would.
    pub fn reason(&self) -> Option<String> {
        match self {
            GgufSpeculation::Available => None,
            GgufSpeculation::NoHead => {
                Some("this file carries no multi-token-prediction head".to_string())
            }
            GgufSpeculation::HeadIgnored { found } => Some(format!(
                "this file's multi-token-prediction head ({found} tensors) is incomplete or \
                 unrecognized, so the install skips it"
            )),
            GgufSpeculation::UnbatchedMatrix { tensor, ggml_type } => Some(format!(
                "{tensor} is {}, which the batched verify has no kernel for; the model \
                 decodes normally, only speculation is unavailable",
                ggml_type_name(*ggml_type).unwrap_or("an unknown type")
            )),
        }
    }
}

/// The resident dtype the walk stores a dense `qwen35` tensor at. Mirrors
/// `transcode::resident_entries` for this family: F32 narrows to raw BF16
/// (the family has no INT8 transcode targets), a K-quant `ssm_out` is
/// re-encoded to Q8_0, and every other type keeps its block type.
/// `predicted_dtypes_match_the_walk` holds the two together.
fn stored_dtype(canonical: &str, ggml_type: u32, arch: &ArchConfig) -> Option<u8> {
    if ggml_type == 0 {
        return Some(crate::resident_writer::DTYPE_BF16);
    }
    if gdn_out_proj_needs_q8_0_requant(canonical, ggml_type, arch) {
        return Some(crate::resident_writer::DTYPE_GGUF_Q8_0);
    }
    dtype_tag_for_ggml_type(ggml_type)
}

/// The install's speculation verdict for this file, or `None` when its
/// family has no GGUF speculation path.
pub fn gguf_speculation(header: &GgufHeader, arch: &ArchConfig) -> Option<GgufSpeculation> {
    if arch.family != ModelFamily::QwenGdnDense || arch.num_experts != 0 {
        return None;
    }
    let num_layers = arch.num_layers as usize;
    let plan = classify(header, arch.family, num_layers).ok()?;
    let mut has_head = false;
    // canonical -> (file name, ggml type), for the resident tensors only.
    let mut resident: BTreeMap<String, (&str, u32)> = BTreeMap::new();
    for name in &plan.resident {
        let Ok(GgufMapping::Resident(canonical)) =
            map_gguf_name_with_head(name, arch.family, num_layers)
        else {
            continue;
        };
        if canonical.starts_with("mtp.") {
            has_head = true;
            continue;
        }
        resident.insert(canonical, (name, header.tensors[*name].ggml_type));
    }
    if !has_head {
        let found = header
            .tensors
            .keys()
            .filter(|n| head_block_index(n, num_layers).is_some())
            .count();
        return Some(if found == 0 {
            GgufSpeculation::NoHead
        } else {
            GgufSpeculation::HeadIgnored { found }
        });
    }
    // Same order as the runtime: layers, then the LM head the verify runs.
    let lm_head = if arch.tie_word_embeddings {
        "language_model.model.embed_tokens.weight"
    } else {
        "language_model.lm_head.weight"
    };
    let names = (0..num_layers)
        .flat_map(|l| {
            MTP_VERIFY_PROJECTIONS
                .iter()
                .map(move |s| format!("language_model.model.layers.{l}.{s}"))
        })
        .chain(std::iter::once(lm_head.to_string()));
    for canonical in names {
        let Some(&(file_name, ggml_type)) = resident.get(&canonical) else {
            continue;
        };
        let batched = stored_dtype(&canonical, ggml_type, arch)
            .is_some_and(|d| MTP_BATCHED_GEMM_DTYPES.contains(&d));
        if !batched {
            return Some(GgufSpeculation::UnbatchedMatrix {
                tensor: file_name.to_string(),
                ggml_type,
            });
        }
    }
    Some(GgufSpeculation::Available)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        build_synthetic_qwen35_dense_gguf, parse_gguf_header, write_gguf_install_streamed,
        MemoryRangeSource, GGUF_DEFAULT_MAX_HEADER_BYTES,
    };

    fn fixture(with_head: bool) -> (Vec<u8>, GgufHeader, ArchConfig) {
        let (bytes, _) = build_synthetic_qwen35_dense_gguf(with_head);
        let header = parse_gguf_header(&bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("parse");
        let arch = crate::arch_from_gguf(&header).expect("arch");
        (bytes, header, arch)
    }

    #[test]
    fn a_k_quant_file_with_a_complete_head_speculates() {
        let (_, header, arch) = fixture(true);
        assert_eq!(
            gguf_speculation(&header, &arch),
            Some(GgufSpeculation::Available)
        );
    }

    #[test]
    fn a_headless_file_says_so() {
        let (_, header, arch) = fixture(false);
        assert_eq!(
            gguf_speculation(&header, &arch),
            Some(GgufSpeculation::NoHead)
        );
    }

    #[test]
    fn a_partial_head_is_reported_as_skipped() {
        let (_, mut header, arch) = fixture(true);
        let dropped = format!("blk.{}.nextn.eh_proj.weight", arch.num_layers);
        assert!(
            header.tensors.remove(&dropped).is_some(),
            "fixture names {dropped}"
        );
        assert_eq!(
            gguf_speculation(&header, &arch),
            Some(GgufSpeculation::HeadIgnored { found: 14 })
        );
    }

    #[test]
    fn an_unbatched_verify_matrix_is_named() {
        let (_, mut header, arch) = fixture(true);
        // Q3_K (ggml 11) has a GEMV but no batched GEMM.
        header
            .tensors
            .get_mut("blk.2.ffn_up.weight")
            .expect("fixture tensor")
            .ggml_type = 11;
        let verdict = gguf_speculation(&header, &arch).expect("qwen35 answers");
        assert_eq!(
            verdict,
            GgufSpeculation::UnbatchedMatrix {
                tensor: "blk.2.ffn_up.weight".to_string(),
                ggml_type: 11
            }
        );
        let reason = verdict.reason().expect("a reason");
        assert!(reason.contains("Q3_K"), "{reason}");
    }

    #[test]
    fn other_families_have_no_prediction() {
        let (_, header, mut arch) = fixture(true);
        arch.family = ModelFamily::Qwen3Dense;
        assert_eq!(gguf_speculation(&header, &arch), None);
    }

    /// `stored_dtype` restates the walk's per-tensor decision, so check it
    /// against what the walk actually writes, tensor by tensor.
    #[test]
    fn predicted_dtypes_match_the_walk() {
        let (bytes, header, _) = fixture(true);
        let dir = std::env::temp_dir().join(format!(
            "turbospark-gguf-speculation-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let arch = write_gguf_install_streamed(
            &dir,
            &header,
            &MemoryRangeSource::new(&bytes),
            "qwen35-speculation",
            |_| {},
        )
        .expect("write install");
        let index = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");
        let plan = classify(&header, arch.family, arch.num_layers as usize).unwrap();
        let mut checked = 0;
        for name in &plan.resident {
            let Ok(GgufMapping::Resident(canonical)) =
                map_gguf_name_with_head(name, arch.family, arch.num_layers as usize)
            else {
                continue;
            };
            let Some(entry) = index.entries.get(&canonical) else {
                continue;
            };
            let ggml_type = header.tensors[*name].ggml_type;
            assert_eq!(
                stored_dtype(&canonical, ggml_type, &arch),
                Some(entry.dtype),
                "{name} -> {canonical} (ggml {ggml_type})"
            );
            checked += 1;
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(checked > 40, "compared only {checked} tensors");
    }
}
