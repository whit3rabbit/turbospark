//! Builds a tiny `qwen3_5` install through the REAL checkpoint repack pipeline.

use model_io::ArchConfig;

pub use super::dense_arch::tiny_qwen_gdn_dense_arch;
pub(crate) use super::dense_arch::*;
use super::dense_tensors::{dflash_drafter_tensors, f16_vector, mtp_head_tensors, packed_triple};
use crate::gemma4_checkpoint::{write_qwen_gdn_dense_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_real::{assemble_safetensors, Tensor};

/// Writes a tiny `qwen3_5` `.gturbo` install and returns the `ArchConfig`
/// needed to open it.
pub fn build_synthetic_qwen_gdn_dense_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_at_bits(dir, vocab_size, num_layers, model_id, 1)
}

/// [`build_synthetic_qwen_gdn_dense_install`] at an explicit affine width:
/// 1 for `prism-ml/Bonsai-27B-mlx-1bit`, 2 for
/// `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (ROADMAP's ternary entry).
pub fn build_synthetic_qwen_gdn_dense_install_at_bits(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, false, false, false, false,
    )
}

/// [`build_synthetic_qwen_gdn_dense_install_at_bits`] with the
/// multi-token-prediction head attached (`docs/MTP_SPECULATIVE.md`, step 1).
///
/// A third entry point rather than a widened signature, following this file's
/// own delegation chain: five callers take the two existing forms and none of
/// them wants a head, so adding a parameter to those would edit five call
/// sites to say `false`.
pub fn build_synthetic_qwen_gdn_dense_install_with_mtp(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, true, false, false, false,
    )
}

/// The same install through the STREAMED writer, which is the one every real
/// checkpoint takes.
///
/// **This entry point exists because its absence shipped a bug.** The head's
/// ingest landed in `orchestrate_gemma4_checkpoint_sharded` alone, every
/// fixture went through that non-streamed path, and
/// `write_gemma4_install_streamed` classified `mtp.*` correctly and then
/// never read it -- so the first real stream that asked for a head wrote a
/// byte-identical HEADLESS install and said nothing. A fixture has to
/// exercise the WRITER the download will use, not just the walk it shares.
pub fn build_synthetic_qwen_gdn_dense_install_with_mtp_streamed(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, true, true, false, false,
    )
}

/// [`build_synthetic_qwen_gdn_dense_install`] with the DFlash2 drafter
/// attached (`docs/DFLASH2.md`), through the NON-streamed writer.
pub fn build_synthetic_qwen_gdn_dense_install_with_dflash(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, false, false, true, false,
    )
}

/// BOTH drafters in one install, which no real checkpoint ships and which
/// exists to make a DISAMBIGUATING fixture possible.
///
/// `crates/cli`'s `resolve_drafter` picks between them off the resident
/// index, and the clause that matters is that an install carrying both keeps
/// the pre-existing MTP behaviour with no DFlash2 note. A dflash-only
/// fixture cannot see that clause: drop it and the dflash-only case still
/// passes, so the mutation survives and the test reads stronger than it is.
/// The install this builds is the only input on which the two rules differ.
pub fn build_synthetic_qwen_gdn_dense_install_with_both_drafters(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, true, false, true, false,
    )
}

/// The same drafter through the STREAMED writer, which is the one the real
/// download takes. Exists for the head's reason verbatim: a drafter arm
/// that only the non-streamed path read would stream a drafterless install
/// and say nothing (see `both_writers_carry_the_mtp_head`'s header).
pub fn build_synthetic_qwen_gdn_dense_install_with_dflash_streamed(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, false, true, true, false,
    )
}

/// [`build_synthetic_qwen_gdn_dense_install`] with a VISION TOWER attached
/// (ROADMAP M-V3), through the NON-streamed writer.
///
/// Depth 2 at hidden 64 (`synthetic_qwen::vision`), which is the tower's real
/// tensor inventory at a size a unit test can digest.
pub fn build_synthetic_qwen_gdn_dense_install_with_vision(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, false, false, false, true,
    )
}

/// The same install through the STREAMED writer, which is the one every real
/// checkpoint takes.
///
/// **This entry point exists for the reason its MTP sibling does, and that
/// reason is a bug that shipped.** The head's ingest landed in the
/// non-streamed walk alone, every fixture went through that path, and the
/// first real stream wrote a byte-identical HEADLESS install with no error.
/// The tower has an arm in both writers from day one, and
/// `both_writers_carry_the_vision_tower` is what keeps it that way: remove
/// either arm and that test reddens.
pub fn build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_qwen_gdn_dense_install_inner(
        dir, vocab_size, num_layers, model_id, bits, false, true, false, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_synthetic_qwen_gdn_dense_install_inner(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
    bits: u32,
    with_mtp: bool,
    streamed: bool,
    with_dflash: bool,
    with_vision: bool,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let mut arch = tiny_qwen_gdn_dense_arch(vocab_size, num_layers);
    // The tower is on the ARCH as well as in the tensor list, because
    // `vision::should_ingest` requires both: the arch is the caller's request
    // and the tensors are the artifact's answer. A fixture that set only one
    // would exercise neither arm of that conjunction.
    if with_vision {
        arch.vision = super::vision::tiny_vision_config();
    }
    let vocab = vocab_size as usize;
    let la = &arch.linear_attention;
    let qkv_dim = la.qkv_dim() as usize;
    let value_dim = la.value_dim() as usize;
    let v_heads = LA_V_HEADS;

    let mut ts: Vec<Tensor> = Vec::new();
    ts.extend(packed_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1,
        bits,
    ));
    ts.extend(packed_triple(
        "language_model.lm_head.weight",
        vocab,
        HIDDEN,
        2,
        bits,
    ));

    for l in 0..num_layers as usize {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
        let is_full = arch.layer_is_full(l);

        for (i, norm) in ["input_layernorm", "post_attention_layernorm"]
            .iter()
            .enumerate()
        {
            ts.push(f16_vector(
                &format!("{p}.{norm}.weight"),
                HIDDEN,
                1.0,
                seed + 40 + i as u64,
            ));
        }

        if is_full {
            ts.extend(packed_triple(
                &format!("{p}.self_attn.q_proj.weight"),
                2 * NUM_HEADS * HEAD_DIM,
                HIDDEN,
                seed + 1,
                bits,
            ));
            for (i, role) in ["k_proj", "v_proj"].iter().enumerate() {
                ts.extend(packed_triple(
                    &format!("{p}.self_attn.{role}.weight"),
                    NUM_KV_HEADS * HEAD_DIM,
                    HIDDEN,
                    seed + 2 + i as u64,
                    bits,
                ));
            }
            ts.extend(packed_triple(
                &format!("{p}.self_attn.o_proj.weight"),
                HIDDEN,
                NUM_HEADS * HEAD_DIM,
                seed + 4,
                bits,
            ));
            for (i, norm) in ["q_norm", "k_norm"].iter().enumerate() {
                ts.push(f16_vector(
                    &format!("{p}.self_attn.{norm}.weight"),
                    HEAD_DIM,
                    1.0,
                    seed + 20 + i as u64,
                ));
            }
        } else {
            for (name, rows, cols, s) in [
                ("in_proj_qkv", qkv_dim, HIDDEN, 1u64),
                ("in_proj_z", value_dim, HIDDEN, 2),
                ("in_proj_a", v_heads, HIDDEN, 3),
                ("in_proj_b", v_heads, HIDDEN, 4),
                ("out_proj", HIDDEN, value_dim, 5),
            ] {
                ts.extend(packed_triple(
                    &format!("{p}.linear_attn.{name}.weight"),
                    rows,
                    cols,
                    seed + s,
                    bits,
                ));
            }
            ts.push(conv1d_weight(
                &format!("{p}.linear_attn.conv1d.weight"),
                qkv_dim,
                LA_CONV_K,
                seed + 10,
            ));
            ts.push(f16_vector(
                &format!("{p}.linear_attn.A_log"),
                v_heads,
                0.0,
                seed + 11,
            ));
            ts.push(f16_vector(
                &format!("{p}.linear_attn.dt_bias"),
                v_heads,
                0.0,
                seed + 12,
            ));
            ts.push(f16_vector(
                &format!("{p}.linear_attn.norm.weight"),
                LA_VALUE_DIM,
                1.0,
                seed + 13,
            ));
        }

        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = if *role == "down_proj" {
                (HIDDEN, INTER)
            } else {
                (INTER, HIDDEN)
            };
            ts.extend(packed_triple(
                &format!("{p}.mlp.{role}.weight"),
                rows,
                cols,
                seed + 60 + i as u64,
                bits,
            ));
        }
    }
    ts.push(f16_vector(
        "language_model.model.norm.weight",
        HIDDEN,
        1.0,
        7,
    ));
    if with_mtp {
        ts.extend(mtp_head_tensors());
    }
    if with_dflash {
        ts.extend(dflash_drafter_tensors(vocab));
    }
    if with_vision {
        ts.extend(super::vision::vision_tower_tensors());
    }

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let quant = Gemma4Quant {
        default_bits: bits,
        group_size: group_for(bits) as u32,
        bits_overrides: std::collections::HashMap::new(),
    };
    if streamed {
        let shards = crate::gemma4_checkpoint::Gemma4Shards::single(&header, &source);
        crate::gemma4_checkpoint::write_qwen_gdn_dense_install_streamed(
            dir,
            &arch,
            model_id,
            &shards,
            &quant,
            |_| {},
        )?;
    } else {
        write_qwen_gdn_dense_install(dir, &arch, model_id, &header, &source, &quant)?;
    }
    Ok(arch)
}

fn conv1d_weight(name: &str, channels: usize, taps: usize, seed: u64) -> Tensor {
    let flat = f16_vector(name, channels * taps, 0.0, seed);
    Tensor {
        name: flat.name,
        dtype: flat.dtype,
        shape: vec![channels as u64, taps as u64, 1],
        bytes: flat.bytes,
    }
}
