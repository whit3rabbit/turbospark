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
        mxfp4: false,
        expert_stride: None,
        // Read through the SAME parser the install writes its manifest with,
        // so a probe and the install it produces cannot disagree about the
        // checkpoint's own window.
        trained_context: repack::trained_context_meta::from_gguf(header),
        sidecars_present: Vec::new(),
        sidecars_missing: Vec::new(),
        chat_template: None,
        verdict: Verdict::Runnable,
        adapter: None,
        speculation: None,
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
            // Types alone said RUNNABLE for an Ornith `qwen3_5` whose Q4_K
            // `ssm_out` the install walk then refused: the V-head reorder
            // cannot split a 256-element block across 128-column heads.
            for reason in v_head_block_refusals(header, &arch) {
                report.refuse(reason);
            }
            if arch.family == model_io::ModelFamily::MiniMaxM2 {
                match repack::minimax_gguf_sizing(header) {
                    Ok(size) => {
                        report.expert_stride = Some(size.max_expert_stride);
                        warnings.push(format!("MiniMax header sizing (bytes): resident {}, expert files {}, complete install allowance {}, eight slots {}, FP16 KV at 8192 {}. These are allocations/storage, not measured footprint.", size.resident_bytes, size.expert_file_bytes, size.install_bytes(), size.eight_slot_bytes, size.kv_8192_bytes));
                    }
                    Err(e) => report.refuse(format!("MiniMax sizing: {e}")),
                }
            }
            // The install re-encodes a GDN `ssm_out` whose block is wider than
            // one V head to Q8_0 (`gdn_out_proj_needs_q8_0_requant`, the SAME
            // predicate the walk uses, so the two cannot drift). Say so: it is
            // not byte-lossless and it grows the install.
            let (mut count, mut from, mut to) = (0usize, 0u64, 0u64);
            for (name, info) in &header.tensors {
                let Ok(repack::GgufMapping::Resident(canonical)) =
                    repack::map_gguf_name(name, family)
                else {
                    continue;
                };
                if repack::gdn_out_proj_needs_q8_0_requant(&canonical, info.ggml_type, &arch) {
                    count += 1;
                    from += info.byte_size(name).unwrap_or(0);
                    // Q8_0: 34 bytes per 32 elements.
                    to += info.dims.iter().product::<u64>() / 32 * 34;
                }
            }
            if count > 0 {
                warnings.push(format!(
                    "{count} linear_attn.out_proj tensor(s) are stored in a block wider than one \
                     V head, so the install dequantizes them, restores the V-head order and \
                     re-encodes them as Q8_0 (not byte-identical to the file; about {} MiB larger)",
                    to.saturating_sub(from) / (1 << 20)
                ));
            }
            report.speculation = repack::gguf_speculation(header, &arch);
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
    // A file that will not install has no install to speculate on.
    if !report.verdict.is_runnable() {
        report.speculation = None;
    }
    report.warnings = warnings;
    report
}

/// The install walk's whole-block V-head refusal, applied to the header so
/// the probe and the walk give one answer (`repack::gguf_v_head_block_split`).
fn v_head_block_refusals(header: &repack::GgufHeader, arch: &ArchConfig) -> Vec<String> {
    header
        .tensors
        .iter()
        .filter_map(|(name, info)| {
            let Ok(repack::GgufMapping::Resident(canonical)) =
                repack::map_gguf_name(name, arch.family)
            else {
                return None;
            };
            let reason = repack::gguf_v_head_block_split(name, &canonical, arch, info.ggml_type)?;
            let type_name = repack::ggml_type_name(info.ggml_type)
                .map(str::to_string)
                .unwrap_or_else(|| format!("type {}", info.ggml_type));
            Some(format!("{name} ({type_name}): {reason}"))
        })
        .collect()
}

/// Per-block-type tensor counts and byte shares, descending by bytes with
/// unsized types last.
fn type_shares(header: &repack::GgufHeader, family: model_io::ModelFamily) -> Vec<TypeShare> {
    // (tensors, bytes, all-transcoded, used-in-routed-experts)
    let mut by_id: std::collections::BTreeMap<u32, (usize, Option<u64>, bool, bool)> =
        std::collections::BTreeMap::new();
    for (name, info) in &header.tensors {
        let source_transcoded = TRANSCODED_GGML_TYPES.contains(&info.ggml_type)
            || (matches!(
                family,
                model_io::ModelFamily::Gemma3 | model_io::ModelFamily::Gemma2
            ) && repack::gemma3_tensor_is_transcoded(info.ggml_type))
            || (family == model_io::ModelFamily::Qwen4Exp
                && matches!(
                    repack::map_gguf_name(name, family),
                    Ok(repack::GgufMapping::Resident(ref canonical))
                        if repack::qwen4exp_tensor_is_transcoded(canonical, info.ggml_type)
                ));
        let slot = by_id
            .entry(info.ggml_type)
            .or_insert((0, Some(0), true, false));
        slot.0 += 1;
        slot.2 &= source_transcoded;
        // A routed expert tensor is the one place a legacy type has no kernel.
        slot.3 |= !source_transcoded && name.contains("_exps");
        // Checked: sizes come off an untrusted header, and a wrapped sum
        // would misreport the share instead of marking it unsized.
        slot.1 = match (slot.1, info.byte_size(name)) {
            (Some(total), Ok(size)) => total.checked_add(size),
            _ => None,
        };
    }
    let mut shares: Vec<TypeShare> = by_id
        .into_iter()
        .map(|(id, (tensors, bytes, transcoded, in_routed))| {
            let name = repack::ggml_type_name(id)
                .map(str::to_string)
                .unwrap_or_else(|| format!("type {id}"));
            let lower = name.to_lowercase();
            // Resident-only types (the legacy blocks) run as attention, FFN,
            // embedding and head tensors, and have no routed-expert kernel, so
            // one that appears in `*_exps` is blocked even though the name is
            // on the executable list.
            let routed_blocked =
                in_routed && model_io::RESIDENT_ONLY_GGUF_TYPES.contains(&lower.as_str());
            let executable = transcoded
                || (model_io::EXECUTABLE_GGUF_TYPES.contains(&lower.as_str()) && !routed_blocked);
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
            total = total.checked_add(info.byte_size(name).ok()?)?;
        }
    }
    (total > 0).then(|| total / arch.num_experts as u64)
}

#[cfg(test)]
mod tests {
    use super::{type_shares, v_head_block_refusals};
    use model_io::ModelFamily;
    use repack::{GgufHeader, GgufTensorInfo};
    use std::collections::BTreeMap;

    fn header(entries: &[(&str, u32)]) -> GgufHeader {
        let tensors = entries
            .iter()
            .enumerate()
            .map(|(index, (name, ggml_type))| {
                let elements = match ggml_type {
                    2 | 3 | 6 | 7 => 32,
                    42 | 40 => 64,
                    41 => 128,
                    34 | 35 => 256,
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

    /// A GDN GGUF with 64-wide V heads and `ssm_out` of the given type. 64 rows
    /// of 256 columns: one Q4_K superblock per row, a quarter of it per head.
    fn gdn_header_with_ssm_out(ggml_type: u32, bytes_per_row: usize) -> GgufHeader {
        let (bytes, _) = repack::GgufBuilder::new()
            .metadata_str("general.architecture", "qwen35moe")
            .metadata_u32("qwen35moe.block_count", 1)
            .metadata_u32("qwen35moe.embedding_length", 64)
            .metadata_u32("qwen35moe.attention.head_count", 4)
            .metadata_u32("qwen35moe.attention.head_count_kv", 2)
            .metadata_u32("qwen35moe.expert_count", 4)
            .metadata_u32("qwen35moe.expert_used_count", 2)
            .metadata_u32("qwen35moe.expert_feed_forward_length", 16)
            .metadata_u32("qwen35moe.full_attention_interval", 4)
            .metadata_u32("qwen35moe.ssm.group_count", 2)
            .metadata_u32("qwen35moe.ssm.time_step_rank", 4)
            .metadata_u32("qwen35moe.ssm.state_size", 64)
            .metadata_u32("qwen35moe.ssm.inner_size", 256)
            .metadata_u32("qwen35moe.ssm.conv_kernel", 2)
            .q8_0_tensor("token_embd.weight", &[64, 128], 1)
            .tensor(
                "blk.0.ssm_out.weight",
                ggml_type,
                &[256, 64],
                vec![0u8; 64 * bytes_per_row],
            )
            .build();
        repack::parse_gguf_header(&bytes, repack::GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap()
    }

    #[test]
    fn a_dense_qwen35_file_reports_whether_it_speculates() {
        let repo = crate::hf::RepoRef::new("owner/name", "main");
        let verdict = |with_head: bool| {
            let (bytes, _) = repack::build_synthetic_qwen35_dense_gguf(with_head);
            let header =
                repack::parse_gguf_header(&bytes, repack::GGUF_DEFAULT_MAX_HEADER_BYTES).unwrap();
            let report = super::evaluate_gguf(&header, &repo, "m.gguf", None);
            assert!(report.verdict.is_runnable(), "{:?}", report.verdict);
            report.speculation
        };
        assert_eq!(verdict(true), Some(repack::GgufSpeculation::Available));
        assert_eq!(verdict(false), Some(repack::GgufSpeculation::NoHead));
    }

    #[test]
    fn a_k_quant_ssm_out_narrower_than_a_block_warns_that_it_is_reencoded() {
        let repo = crate::hf::RepoRef::new("owner/name", "main");
        let warned = |ggml_type: u32, bytes_per_row: usize| {
            super::evaluate_gguf(
                &gdn_header_with_ssm_out(ggml_type, bytes_per_row),
                &repo,
                "m.gguf",
                None,
            )
            .warnings
            .iter()
            .any(|w| w.contains("re-encodes them as Q8_0"))
        };
        // Q4_K: 144 bytes per 256 elements, wider than a 64-wide head.
        assert!(warned(12, 144), "Q4_K ssm_out must be flagged");
        // Q8_0: 32-element blocks divide a head, so it is a pure byte move.
        assert!(
            !warned(8, 256 / 32 * 34),
            "Q8_0 ssm_out must not be flagged"
        );
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

        assert!(
            share.executable,
            "routed Q4_0 runs on its own routed kernels"
        );
        assert!(
            !share.transcoded,
            "only resident tensors take the transcode path"
        );
    }

    /// Q1_0 (41), TQ2_0 (35), TQ1_0 (34) and NVFP4 (40) have a resident GEMV
    /// and a routed pair, so the probe calls them executable on an attention
    /// tensor AND on a routed expert.
    #[test]
    fn ext_block_types_are_executable_resident_and_routed() {
        for (id, name) in [(41, "Q1_0"), (35, "TQ2_0"), (34, "TQ1_0"), (40, "NVFP4")] {
            let resident = header(&[("blk.0.attn_q.weight", id)]);
            let shares = type_shares(&resident, ModelFamily::Qwen4Exp);
            assert!(
                share(&shares, name).executable,
                "{name} resident has a kernel"
            );

            let routed = header(&[("blk.0.ffn_gate_exps.weight", id)]);
            let shares = type_shares(&routed, ModelFamily::Qwen4Exp);
            assert!(
                share(&shares, name).executable,
                "{name} routed has a kernel"
            );
        }
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

    /// Two F32 tensors of 2^61 elements each are 2^63 bytes apiece: each
    /// sizes on its own, but their sum passes u64::MAX. The header is
    /// attacker-authored and nothing upstream bounds extents against the
    /// file, so the sum must refuse rather than wrap (release) or panic
    /// (debug).
    fn overflowing_header(names: &[&str]) -> GgufHeader {
        let tensors = names
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    GgufTensorInfo {
                        ggml_type: 0,
                        dims: vec![1u64 << 61],
                        offset: 0,
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

    #[test]
    fn type_share_bytes_that_overflow_u64_are_reported_unsized() {
        let header = overflowing_header(&["blk.0.attn_q.weight", "blk.1.attn_q.weight"]);
        let shares = type_shares(&header, ModelFamily::Gemma4);
        assert_eq!(share(&shares, "F32").bytes, None);
    }

    #[test]
    fn an_expert_stride_that_overflows_u64_is_not_reported() {
        let (bytes, _) = repack::build_synthetic_gemma4_gguf(repack::SyntheticGgufShape::default());
        let real = repack::parse_gguf_header(&bytes, repack::GGUF_DEFAULT_MAX_HEADER_BYTES)
            .expect("fixture parses");
        let arch = repack::arch_from_gguf(&real).expect("fixture arch");
        assert!(arch.num_experts > 0, "fixture must be MoE to reach the sum");
        let header =
            overflowing_header(&["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"]);
        assert_eq!(super::gguf_expert_stride(&header, &arch), None);
    }

    #[test]
    fn resident_legacy_types_run_on_their_own_kernels_not_by_transcoding() {
        // A resident Q4_0/Q4_1/Q5_0/Q5_1 projection has a GEMV and an embedding
        // lookup. It is executable, and it is NOT transcoded: only Qwen4Exp's
        // source tensors take that path.
        let header = header(&[
            ("blk.0.attn_q.weight", 2),
            ("blk.0.attn_k.weight", 3),
            ("blk.0.attn_v.weight", 6),
            ("blk.0.attn_output.weight", 7),
        ]);
        let shares = type_shares(&header, ModelFamily::QwenGdnMoe);
        for name in ["Q4_0", "Q4_1", "Q5_0", "Q5_1"] {
            let share = share(&shares, name);
            assert!(share.executable, "{name} has resident kernels");
            assert!(!share.transcoded, "{name} is not converted at repack");
        }
    }

    #[test]
    fn legacy_types_in_routed_experts_are_executable_for_every_family() {
        // The legacy blocks have routed pairs now, so a Mixtral-style Q4_0
        // file is executable end to end, experts included.
        for ty in [2u32, 3, 6, 7] {
            let header = header(&[
                ("blk.0.attn_q.weight", ty),
                ("blk.0.ffn_gate_exps.weight", ty),
            ]);
            let shares = type_shares(&header, ModelFamily::QwenGdnMoe);
            let name = repack::ggml_type_name(ty).unwrap();
            assert!(
                share(&shares, name).executable,
                "{name} experts have a routed kernel"
            );
        }
    }

    #[test]
    fn gdn_superblock_ssm_out_is_refused_by_the_probe() {
        let mut arch = model_io::known_architecture(ModelFamily::QwenGdnDense);
        arch.linear_attention.value_head_dim = 128;
        let mut header = header(&[]);
        for (name, ggml_type) in [("blk.0.ssm_out.weight", 12u32), ("blk.1.ssm_out.weight", 8)] {
            header.tensors.insert(
                name.to_string(),
                GgufTensorInfo {
                    ggml_type,
                    dims: vec![256, 4],
                    offset: 0,
                },
            );
        }
        let refusals = v_head_block_refusals(&header, &arch);
        assert_eq!(refusals.len(), 1, "{refusals:?}");
        assert!(
            refusals[0].starts_with("blk.0.ssm_out.weight (Q4_K)"),
            "{refusals:?}"
        );
    }
}
