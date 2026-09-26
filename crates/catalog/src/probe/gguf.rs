//! The GGUF half of the probe: architecture, block types, expert stride.

use model_io::ArchConfig;
use repack::ArchSupport;

use super::{ProbeReport, TypeShare, Verdict};
use crate::hf::{Client, RepoRef};

/// ggml type ids the repack walk transcodes rather than dispatches.
///
/// **Checking these against `EXECUTABLE_GGUF_TYPES` marks every real
/// candidate blocked**, which is exactly what
/// `scopes_the_dense_llama_candidates`' first run did
/// (`crates/repack/CLAUDE.md` Gotcha 5). They are narrowed at repack time and
/// reach no dispatch at all.
const TRANSCODED_GGML_TYPES: [u32; 3] = [0, 1, 30]; // F32, F16, BF16

pub(super) fn probe_gguf(
    client: &Client,
    repo: &RepoRef,
    file: &str,
) -> Result<ProbeReport, String> {
    let source = crate::gguf_source::load(client, repo, file, None, None, None)?;
    Ok(evaluate_gguf(
        &source.header,
        repo,
        file,
        Some(source.bytes),
    ))
}

/// Every GGUF gate, against a header that is already in hand.
///
/// **Split out from the fetching wrapper so it can be tested with no
/// network**, against `repack::build_synthetic_gemma4_gguf`'s fixtures. That
/// matters more here than it looks: the gates are where the decisions are,
/// and the refusal paths (a planned architecture, an unexecutable block type,
/// an unsizeable one) are exactly the paths a live probe of a curated row
/// never takes.
pub fn evaluate_gguf(
    header: &repack::GgufHeader,
    repo: &RepoRef,
    file: &str,
    download_bytes: Option<u64>,
) -> ProbeReport {
    let architecture = header.architecture().map(str::to_string);
    let mut warnings = Vec::new();
    let mut report = ProbeReport {
        repo: repo.clone(),
        kind: crate::SourceKind::Gguf,
        file: Some(file.to_string()),
        download_bytes,
        architecture: architecture.clone(),
        family: None,
        arch: None,
        types: Vec::new(),
        affine: None,
        expert_stride: None,
        // Read through the SAME parser the install writes its manifest with,
        // so a probe and the install it produces cannot disagree about the
        // checkpoint's own window.
        trained_context: repack::trained_context_meta::from_gguf(header),
        sidecars_present: Vec::new(),
        sidecars_missing: Vec::new(),
        chat_template: None,
        verdict: Verdict::Runnable,
        warnings: Vec::new(),
    };

    let Some(architecture) = architecture else {
        report.refuse(
            "the file declares no general.architecture, so it is not a GGUF this parser \
             can place"
                .to_string(),
        );
        return report;
    };

    let family = match repack::gguf_arch_support(&architecture) {
        Some(ArchSupport::Supported(family)) => family,
        _ => {
            // `describe_gguf_architecture` distinguishes recognized-but-
            // unported (with the clause naming what it needs) from unknown,
            // and carries the bring-up checklist either way. Reusing it keeps
            // one wording rather than two that drift.
            report.refuse(repack::describe_gguf_architecture(&architecture));
            return report;
        }
    };
    report.family = Some(family);

    report.types = type_shares(header, family);
    match repack::arch_from_gguf(header) {
        Ok(arch) => {
            report.expert_stride = gguf_expert_stride(header, &arch);
            if arch.family == model_io::ModelFamily::MiniMaxM2 {
                match repack::minimax_gguf_sizing(header) {
                    Ok(size) => {
                        report.expert_stride = Some(size.max_expert_stride);
                        warnings.push(format!("MiniMax header sizing (bytes): resident {}, expert files {}, complete install allowance {}, eight slots {}, FP16 KV at 8192 {}. These are allocations/storage, not measured footprint.", size.resident_bytes, size.expert_file_bytes, size.install_bytes(), size.eight_slot_bytes, size.kv_8192_bytes));
                    }
                    Err(e) => report.refuse(format!("MiniMax sizing: {e}")),
                }
            }
            report.arch = Some(arch);
        }
        Err(e) => {
            report.refuse(format!("deriving an ArchConfig from the header: {e}"));
            return report;
        }
    }

    let blocked: Vec<&TypeShare> = report.types.iter().filter(|t| !t.executable).collect();
    if !blocked.is_empty() {
        let names: Vec<&str> = blocked.iter().map(|t| t.name.as_str()).collect();
        report.refuse(format!(
            "no kernels for block type(s) {}. This port dispatches {} \
             (F32/F16/BF16 are transcoded at repack and need none). Adding one means \
             landing a resident GEMV, an embedding lookup and a routed-expert pair; \
             see docs/NEW_MODEL.md.",
            names.join(", "),
            model_io::EXECUTABLE_GGUF_TYPES.join(", ")
        ));
    }

    if let Some(share) = report.types.iter().find(|t| t.bytes.is_none()) {
        warnings.push(format!(
            "{} could not be sized, so the byte shares below exclude it and understate it",
            share.name
        ));
    }
    report.warnings = warnings;
    report
}

/// Per-block-type tensor counts and byte shares, descending by bytes with
/// unsized types last.
fn type_shares(header: &repack::GgufHeader, family: model_io::ModelFamily) -> Vec<TypeShare> {
    let mut by_id: std::collections::BTreeMap<u32, (usize, Option<u64>, bool)> =
        std::collections::BTreeMap::new();
    for (name, info) in &header.tensors {
        let source_transcoded = TRANSCODED_GGML_TYPES.contains(&info.ggml_type)
            || (family == model_io::ModelFamily::Qwen4Exp
                && matches!(
                    repack::map_gguf_name(name, family),
                    Ok(repack::GgufMapping::Resident(ref canonical))
                        if repack::qwen4exp_tensor_is_transcoded(canonical, info.ggml_type)
                ));
        let slot = by_id.entry(info.ggml_type).or_insert((0, Some(0), true));
        slot.0 += 1;
        slot.2 &= source_transcoded;
        match (slot.1, info.byte_size(name)) {
            (Some(total), Ok(size)) => slot.1 = Some(total + size),
            _ => slot.1 = None,
        }
    }
    let mut shares: Vec<TypeShare> = by_id
        .into_iter()
        .map(|(id, (tensors, bytes, transcoded))| {
            let name = repack::ggml_type_name(id)
                .map(str::to_string)
                .unwrap_or_else(|| format!("type {id}"));
            let executable = transcoded
                || model_io::EXECUTABLE_GGUF_TYPES.contains(&name.to_lowercase().as_str());
            TypeShare {
                name,
                tensors,
                bytes,
                executable,
                transcoded,
            }
        })
        .collect();
    shares.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.name.cmp(&b.name)));
    shares
}

/// Bytes of ONE routed expert, from layer 0's `ffn_*_exps` tensors.
///
/// This is the number Gotcha 36 is about, and it is one multiplication off
/// the header rather than something a download has to reveal.
fn gguf_expert_stride(header: &repack::GgufHeader, arch: &ArchConfig) -> Option<u64> {
    if arch.num_experts <= 0 {
        return None;
    }
    let mut total = 0u64;
    for (name, info) in &header.tensors {
        if name.starts_with("blk.0.") && name.contains("_exps") {
            total += info.byte_size(name).ok()?;
        }
    }
    (total > 0).then(|| total / arch.num_experts as u64)
}

#[cfg(test)]
mod tests {
    use super::type_shares;
    use model_io::ModelFamily;
    use repack::{GgufHeader, GgufTensorInfo};
    use std::collections::BTreeMap;

    fn header(entries: &[(&str, u32)]) -> GgufHeader {
        let tensors = entries
            .iter()
            .enumerate()
            .map(|(index, (name, ggml_type))| {
                let elements = match ggml_type {
                    2 | 6 => 32,
                    42 => 64,
                    _ => panic!("unexpected fixture type {ggml_type}"),
                };
                (
                    (*name).to_string(),
                    GgufTensorInfo {
                        ggml_type: *ggml_type,
                        dims: vec![elements],
                        offset: (index * 32) as u64,
                    },
                )
            })
            .collect();
        GgufHeader {
            version: 3,
            metadata: BTreeMap::new(),
            tensors,
            alignment: 32,
            data_region_start: 0,
        }
    }

    fn share<'a>(shares: &'a [super::TypeShare], name: &str) -> &'a super::TypeShare {
        shares
            .iter()
            .find(|share| share.name == name)
            .unwrap_or_else(|| panic!("missing {name} type share"))
    }

    #[test]
    fn qwen4exp_resident_q4_and_q5_are_reported_as_transcoded() {
        let header = header(&[
            ("blk.0.ffn_down_shexp.weight", 2),
            ("blk.1.ffn_down_shexp.weight", 6),
        ]);
        let shares = type_shares(&header, ModelFamily::Qwen4Exp);

        for name in ["Q4_0", "Q5_0"] {
            let share = share(&shares, name);
            assert!(share.executable, "{name} is converted before runtime");
            assert!(share.transcoded, "{name} has no runtime kernel");
        }
    }

    #[test]
    fn qwen4exp_q4_is_not_transcoded_when_it_occurs_in_routed_experts() {
        let header = header(&[
            ("blk.0.ffn_down_shexp.weight", 2),
            ("blk.0.ffn_gate_exps.weight", 2),
        ]);
        let shares = type_shares(&header, ModelFamily::Qwen4Exp);
        let share = share(&shares, "Q4_0");

        assert!(!share.executable, "the routed Q4_0 source has no kernel");
        assert!(
            !share.transcoded,
            "only resident tensors take the transcode path"
        );
    }

    #[test]
    fn qwen4exp_q2_keeps_its_kernel_classification_for_mixed_resident_and_routed_use() {
        let header = header(&[
            ("blk.0.attn_q.weight", 42),
            ("blk.0.ffn_gate_exps.weight", 42),
        ]);
        let shares = type_shares(&header, ModelFamily::Qwen4Exp);
        let share = share(&shares, "Q2_0");

        assert!(share.executable, "Q2_0 routed experts have a kernel pair");
        assert!(!share.transcoded, "the type share includes routed weights");
    }

    #[test]
    fn q4_is_not_implicitly_transcoded_for_other_families() {
        let header = header(&[("blk.0.attn_q.weight", 2)]);
        let shares = type_shares(&header, ModelFamily::QwenGdnMoe);
        let share = share(&shares, "Q4_0");

        assert!(!share.executable);
        assert!(!share.transcoded);
    }
}
