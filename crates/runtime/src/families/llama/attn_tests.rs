use super::*;
use crate::real_forward::RealForwardRunner;
use half::f16;
use turbospark_repack::build_synthetic_dense_llama_install;

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const LIVE_ROWS: usize = 3;

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
        let position = row;
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
    let pass = staged.context.begin_pass_labeled("llama staged rows");
    let batch = encode_attention_inputs_batch(
        &mut staged.context,
        &pass,
        &staged.weights,
        &staged.index,
        &staged.arch,
        staged.real_llama.as_ref().expect("Llama state"),
        &staged.scratch,
        &staged.kv,
        layer,
        0,
        &rope_positions,
    )
    .unwrap();
    pass.commit_and_wait();

    assert_eq!(batch.first_query_position, 0);
    assert_eq!(batch.row_count, LIVE_ROWS);
    for (row, (expected_q, expected_k, expected_v)) in reference_qkv.iter().enumerate() {
        let q_offset = row * q_dim * 2;
        let (k_buffer, k_offset) = staged.kv.k_slot(layer, row);
        let (v_buffer, v_offset) = staged.kv.v_slot(layer, row);
        assert_eq!(
            half_bits(&gpu::read_buffer_f16(batch.query, q_offset, q_dim)),
            half_bits(expected_q),
            "query row {row} differs from the existing one-row path"
        );
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
    }
}
