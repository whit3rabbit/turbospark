use super::*;
use crate::producer::{ChunkedPrefillRunner, LogitProducer};
use crate::real_forward::RealForwardRunner;
use half::f16;
use turbospark_repack::build_synthetic_dense_llama_install;

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const LIVE_ROWS: usize = 3;
const FIRST_POSITION: usize = 2;

fn open_runner(tag: &str) -> RealForwardRunner {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-llama-attn-stage-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    build_synthetic_dense_llama_install(&dir, VOCAB, LAYERS, "tiny-mistral")
        .expect("synthetic dense Llama install builds");
    let arch = turbospark_repack::peek_manifest_arch(&dir)
        .expect("synthetic dense Llama architecture peeks");
    RealForwardRunner::open(&dir, arch).expect("synthetic dense Llama install opens")
}

fn open_runner_with_kv_quant(tag: &str, kv_quant: model_io::KvQuant) -> RealForwardRunner {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-llama-attn-stage-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let arch = build_synthetic_dense_llama_install(&dir, VOCAB, LAYERS, "tiny-mistral")
        .expect("synthetic dense Llama install builds");
    RealForwardRunner::open_with_kv_quant(
        &dir,
        arch,
        64,
        model_io::ExpertCacheSlots::Fixed(4),
        crate::families::qwen::DraftPolicies::off(),
        crate::steering::SteeringPolicy::off(),
        1,
        kv_quant,
    )
    .expect("synthetic dense Llama install opens with KV mode")
}

fn write_input_rows(runner: &RealForwardRunner, row_count: usize) {
    let hidden = runner.arch.hidden_size as usize;
    for row in 0..row_count {
        let bytes: Vec<u8> = (0..hidden)
            .flat_map(|column| {
                let value = ((row * 7 + column * 3) as f32 - 11.0) / 17.0;
                f16::from_f32(value).to_bits().to_le_bytes()
            })
            .collect();
        gpu::write_buffer_bytes(&runner.scratch.x, row * hidden * 2, &bytes);
    }
}

fn half_bits(values: &[f16]) -> Vec<u16> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn sequential_logits(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<u16> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in tokens.iter().enumerate() {
        if position + 1 == tokens.len() {
            runner.produce(token, position, &mut logits).unwrap();
        } else {
            runner
                .produce_prefill(token, position, &mut logits)
                .unwrap();
        }
    }
    half_bits(&logits)
}

#[test]
fn eligible_multi_row_chunk_uses_one_batch_dispatch_per_layer_and_preserves_logits() {
    let tokens = [5, 9, 2];
    let mut reference = open_runner("batch-route-reference");
    let expected = sequential_logits(&mut reference, &tokens);

    let mut candidate = open_runner("batch-route-candidate");
    let mut actual = vec![f16::from_f32(0.0); VOCAB as usize];
    crate::families::llama::prefill::reset_batch_attention_dispatch_count();
    candidate.prefill_chunk(&tokens, 0, &mut actual).unwrap();

    assert_eq!(
        crate::families::llama::prefill::batch_attention_dispatch_count(),
        LAYERS as usize,
        "an eligible chunk dispatches one batch attention call per layer"
    );
    assert_eq!(
        half_bits(&actual),
        expected,
        "batched attention must preserve row order through output projection and FFN"
    );
}

#[test]
fn batch_attention_route_requires_multi_row_dense_full_unquantized_llama() {
    let runner = open_runner("route-eligibility");
    let eligible = crate::families::llama::prefill::batch_attention_route_eligible(
        &runner.arch,
        &runner.kv,
        2,
    );
    assert!(
        eligible,
        "dense full-attention Llama with FP16 KV is eligible"
    );
    assert!(
        !crate::families::llama::prefill::batch_attention_route_eligible(
            &runner.arch,
            &runner.kv,
            1,
        ),
        "single-row chunks keep the existing attention route"
    );

    let mut qwen = runner.arch.clone();
    qwen.family = model_io::ModelFamily::Qwen3Dense;
    assert!(
        !crate::families::llama::prefill::batch_attention_route_eligible(&qwen, &runner.kv, 2),
        "the shared Llama state type does not make Qwen eligible"
    );

    let mut moe = runner.arch.clone();
    moe.num_experts = 1;
    assert!(
        !crate::families::llama::prefill::batch_attention_route_eligible(&moe, &runner.kv, 2),
        "the Llama family enum also covers MoE configurations"
    );

    let mut partial_attention = runner.arch.clone();
    partial_attention.full_attention_layer_mask[0] = 0;
    assert!(
        !crate::families::llama::prefill::batch_attention_route_eligible(
            &partial_attention,
            &runner.kv,
            2,
        ),
        "every layer must use full attention"
    );

    let quantized = open_runner_with_kv_quant(
        "route-quantized",
        model_io::KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    assert!(
        !crate::families::llama::prefill::batch_attention_route_eligible(
            &quantized.arch,
            &quantized.kv,
            2,
        ),
        "any quantized KV layer keeps the request on the existing path"
    );
}

#[test]
fn quantized_multi_row_chunk_keeps_sequential_logits_and_skips_batch_dispatch() {
    let tokens = [5, 9, 2];
    let mut reference = open_runner_with_kv_quant(
        "quantized-reference",
        model_io::KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    let expected = sequential_logits(&mut reference, &tokens);

    let mut candidate = open_runner_with_kv_quant(
        "quantized-candidate",
        model_io::KvQuant::TurboQuant {
            k_bits: 3,
            v_bits: 4,
        },
    );
    let mut actual = vec![f16::from_f32(0.0); VOCAB as usize];
    crate::families::llama::prefill::reset_batch_attention_dispatch_count();
    candidate.prefill_chunk(&tokens, 0, &mut actual).unwrap();

    assert_eq!(
        crate::families::llama::prefill::batch_attention_dispatch_count(),
        0,
        "KV-quantized multi-row requests must not invoke batch attention"
    );
    assert_eq!(half_bits(&actual), expected);
}

#[test]
fn single_row_chunk_keeps_the_existing_attention_route() {
    let tokens = [5];
    let mut reference = open_runner("single-row-reference");
    let expected = sequential_logits(&mut reference, &tokens);

    let mut candidate = open_runner("single-row-candidate");
    let mut actual = vec![f16::from_f32(0.0); VOCAB as usize];
    crate::families::llama::prefill::reset_batch_attention_dispatch_count();
    candidate.prefill_chunk(&tokens, 0, &mut actual).unwrap();

    assert_eq!(
        crate::families::llama::prefill::batch_attention_dispatch_count(),
        0,
        "single-row chunks must keep the existing one-row attention helper"
    );
    assert_eq!(half_bits(&actual), expected);
}

#[test]
fn batch_staging_matches_single_row_qkv_for_each_live_position() {
    let mut reference = open_runner("reference");
    let mut staged = open_runner("staged");
    write_input_rows(&reference, LIVE_ROWS);
    write_input_rows(&staged, LIVE_ROWS);

    let layer = 0;
    let hidden = reference.arch.hidden_size as usize;
    let q_dim = reference.arch.num_heads as usize * reference.arch.full_head_dim as usize;
    let kv_dim = reference.arch.num_full_kv_heads as usize * reference.arch.full_head_dim as usize;
    let llama = reference.real_llama.as_ref().expect("Llama state");
    let mut reference_qkv = Vec::with_capacity(LIVE_ROWS);

    for row in 0..LIVE_ROWS {
        let position = FIRST_POSITION + row;
        let pass = reference.context.begin_pass_labeled("llama row reference");
        let input_norm = crate::real_forward_utils::norm_view(
            &reference.weights,
            &reference.index,
            &layer_tensor(layer, "input_layernorm.weight"),
            hidden,
        )
        .unwrap();
        gpu::encode_rms_norm_bf16w(
            &mut reference.context,
            &pass,
            (&reference.scratch.x, (row * hidden * 2) as u64),
            input_norm,
            (&reference.scratch.normed, 0),
            hidden as u32,
            llama.rms_eps,
        )
        .unwrap();
        encode_attention_block(
            &mut reference.context,
            &pass,
            &reference.weights,
            &reference.index,
            &reference.arch,
            llama,
            &reference.scratch,
            &reference.kv,
            layer,
            position,
            crate::vision::RopePosition::Sequential,
        )
        .unwrap();
        pass.commit_and_wait();

        let (k_buffer, k_offset) = reference.kv.k_slot(layer, position);
        let (v_buffer, v_offset) = reference.kv.v_slot(layer, position);
        reference_qkv.push((
            gpu::read_buffer_f16(&reference.scratch.q, 0, q_dim),
            gpu::read_buffer_f16(k_buffer, k_offset, kv_dim),
            gpu::read_buffer_f16(v_buffer, v_offset, kv_dim),
        ));
    }

    let rope_positions = vec![crate::vision::RopePosition::Sequential; LIVE_ROWS];
    let batch = crate::families::llama::prefill::stage_attention_inputs_batch(
        &mut staged.context,
        &staged.weights,
        &staged.index,
        &staged.arch,
        staged.real_llama.as_ref().expect("Llama state"),
        &staged.scratch,
        &staged.kv,
        layer,
        FIRST_POSITION,
        &rope_positions,
    )
    .unwrap();

    assert_eq!(batch.first_query_position, FIRST_POSITION);
    assert_eq!(batch.row_count, LIVE_ROWS);
    for (row, (expected_q, expected_k, expected_v)) in reference_qkv.iter().enumerate() {
        let q_offset = row * q_dim * 2;
        let position = FIRST_POSITION + row;
        let (k_buffer, k_offset) = staged.kv.k_slot(layer, position);
        let (v_buffer, v_offset) = staged.kv.v_slot(layer, position);
        assert_eq!(
            half_bits(&gpu::read_buffer_f16(k_buffer, k_offset, kv_dim)),
            half_bits(expected_k),
            "key row {row} differs from the existing one-row path"
        );
        assert_eq!(
            half_bits(&gpu::read_buffer_f16(v_buffer, v_offset, kv_dim)),
            half_bits(expected_v),
            "value row {row} differs from the existing one-row path"
        );
        assert_eq!(
            half_bits(&gpu::read_buffer_f16(batch.query, q_offset, q_dim)),
            half_bits(expected_q),
            "query row {row} differs from the existing one-row path"
        );
    }
}
