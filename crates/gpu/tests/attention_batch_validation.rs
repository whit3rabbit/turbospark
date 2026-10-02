//! Verifies malformed batch inputs are rejected before either Metal pass is encoded.
#![cfg(target_os = "macos")]

use turbospark_gpu::{
    autorelease_pool, encode_attention_decode_batch, BatchAttentionInputContract,
    BatchAttentionKvFormat, BatchAttentionKvLayout, BatchAttentionScratch,
    BatchAttentionScratchLayout, GpuError, MetalContext,
};

const HEAD_DIM: u32 = 4;
const NUM_Q_HEADS: u32 = 1;
const NUM_KV_HEADS: u32 = 1;
const LIVE_ROWS: usize = 2;
const FIRST_QUERY_POSITION: u32 = 31;
const CANARY: u8 = 0x5a;

#[derive(Clone, Copy, Debug)]
enum InvalidInput {
    PositionOverflow,
    ShortQ,
    ShortK,
    ShortV,
    ShortOutput,
    ShortRowPlan,
    ShortPartialM,
    ShortPartialD,
    ShortPartialO,
}

impl InvalidInput {
    const ALL: [Self; 9] = [
        Self::PositionOverflow,
        Self::ShortQ,
        Self::ShortK,
        Self::ShortV,
        Self::ShortOutput,
        Self::ShortRowPlan,
        Self::ShortPartialM,
        Self::ShortPartialD,
        Self::ShortPartialO,
    ];
}

#[test]
fn public_encoder_refuses_invalid_batch_inputs_without_writing_or_dispatching() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let layout = BatchAttentionScratchLayout::new(NUM_Q_HEADS as usize, HEAD_DIM as usize)
        .expect("batch layout");
    let k_v_bytes = (FIRST_QUERY_POSITION as u64 + LIVE_ROWS as u64)
        * NUM_KV_HEADS as u64
        * HEAD_DIM as u64
        * 2;

    for invalid in InvalidInput::ALL {
        autorelease_pool(|| {
            let q_bytes = layout.q_buffer_bytes() as u64;
            let output_bytes = layout.output_buffer_bytes() as u64;
            let row_plan_bytes = layout.row_plan_bytes() as u64;
            let q = context.new_output_buffer(if matches!(invalid, InvalidInput::ShortQ) {
                q_bytes - 1
            } else {
                q_bytes
            });
            let k = context.new_output_buffer(if matches!(invalid, InvalidInput::ShortK) {
                k_v_bytes - 1
            } else {
                k_v_bytes
            });
            let v = context.new_output_buffer(if matches!(invalid, InvalidInput::ShortV) {
                k_v_bytes - 1
            } else {
                k_v_bytes
            });
            let output =
                context.new_output_buffer(if matches!(invalid, InvalidInput::ShortOutput) {
                    output_bytes - 1
                } else {
                    output_bytes
                });
            let mut scratch = BatchAttentionScratch::new(&context, layout).expect("batch scratch");

            match invalid {
                InvalidInput::ShortRowPlan => {
                    scratch.row_plan = context.new_output_buffer(row_plan_bytes - 1);
                }
                InvalidInput::ShortPartialM => {
                    scratch.m = context.new_output_buffer(scratch.m.length() - 1);
                }
                InvalidInput::ShortPartialD => {
                    scratch.d = context.new_output_buffer(scratch.d.length() - 1);
                }
                InvalidInput::ShortPartialO => {
                    scratch.o = context.new_output_buffer(scratch.o.length() - 1);
                }
                _ => {}
            }

            for buffer in [
                &q,
                &k,
                &v,
                &output,
                &scratch.row_plan,
                &scratch.m,
                &scratch.d,
                &scratch.o,
            ] {
                fill_canary(buffer);
            }
            let before = snapshot_writable_buffers(&output, &scratch);
            let first_query_position = if matches!(invalid, InvalidInput::PositionOverflow) {
                u32::MAX
            } else {
                FIRST_QUERY_POSITION
            };

            let pass = context.begin_pass();
            let result = encode_attention_decode_batch(
                &mut context,
                &pass,
                (&q, 0),
                &k,
                &v,
                &scratch,
                (&output, 0),
                linear_fp16_contract(),
                first_query_position,
                LIVE_ROWS,
                HEAD_DIM,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                0.5,
            );
            assert!(
                matches!(result, Err(GpuError::InvalidInput(_))),
                "{invalid:?} should return InvalidInput, got {result:?}"
            );

            // Committing an empty pass is safe and proves the error path did not
            // leave either attention dispatch queued in the caller's encoder.
            pass.commit_and_wait();
            assert_eq!(
                snapshot_writable_buffers(&output, &scratch),
                before,
                "{invalid:?} changed output, row-plan, or partial-state canaries"
            );
        });
    }
}

fn linear_fp16_contract() -> BatchAttentionInputContract {
    BatchAttentionInputContract {
        kv_layout: BatchAttentionKvLayout::Linear,
        kv_format: BatchAttentionKvFormat::Fp16,
        kv_start: 0,
        ring_capacity: 0,
        sink_count: 0,
    }
}

fn fill_canary(buffer: &metal::Buffer) {
    unsafe {
        std::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), buffer.length() as usize)
            .fill(CANARY);
    }
}

fn snapshot(buffer: &metal::Buffer) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u8>(), buffer.length() as usize)
            .to_vec()
    }
}

fn snapshot_writable_buffers(
    output: &metal::Buffer,
    scratch: &BatchAttentionScratch,
) -> [Vec<u8>; 5] {
    [
        snapshot(output),
        snapshot(&scratch.row_plan),
        snapshot(&scratch.m),
        snapshot(&scratch.d),
        snapshot(&scratch.o),
    ]
}
