use super::{
    attention_constants_key, attention_function_constants, build_batch_attention_plan, chunks_for,
    validate_batch_row_plans, BatchAttentionBufferLengths, BatchAttentionInputContract,
    BatchAttentionKvFormat, BatchAttentionKvLayout, RowChunkPlan, MAX_CHUNKS,
    MIN_POSITIONS_PER_CHUNK, SOURCE, THREADS_PER_GROUP,
};

/// The parity tests exercise the chunked path but cannot pin this
/// mapping: a `chunks_for` that always returned 1 would still match
/// the CPU reference, just slowly. These are the boundaries that
/// matter -- never 0 (the kernel takes `% num_chunks`), never above
/// the scratch's capacity, and 1 while the range is short enough that
/// splitting would cost more than it buys.
#[test]
fn chunk_count_stays_within_its_bounds() {
    for range in [1u32, 2, 15, 16, 31, 32, 255, 256, 1024, 4096, u32::MAX] {
        let chunks = chunks_for(range);
        assert!(chunks >= 1, "range {range}: {chunks}");
        assert!(chunks <= MAX_CHUNKS, "range {range}: {chunks}");
        assert!(chunks.is_power_of_two(), "range {range}: {chunks}");
        assert!(
            chunks == 1 || chunks * MIN_POSITIONS_PER_CHUNK <= range,
            "range {range} split {chunks} ways leaves chunks under {MIN_POSITIONS_PER_CHUNK}"
        );
    }
}

#[test]
fn short_ranges_stay_unsplit_and_long_ones_saturate() {
    assert_eq!(chunks_for(0), 1);
    assert_eq!(chunks_for(MIN_POSITIONS_PER_CHUNK * 2 - 1), 1);
    assert_eq!(chunks_for(MIN_POSITIONS_PER_CHUNK * 2), 2);
    assert_eq!(chunks_for(MIN_POSITIONS_PER_CHUNK * MAX_CHUNKS), MAX_CHUNKS);
    assert_eq!(chunks_for(u32::MAX), MAX_CHUNKS);
}

/// Reflect the compiled batch partial pipeline at its maximum four-row tile
/// and head dimension. The head counts and row plan use a supported GQA
/// shape; the production pipeline keeps those dimensions as runtime inputs,
/// while scale and NC are the function constants used by the encoder.
#[test]
fn batch_partial_pipeline_fits_supported_device_resources() {
    let (head_dim, num_q_heads, num_kv_heads, live_rows) = (512u32, 32u32, 8u32, 16u32);
    let input_contract = BatchAttentionInputContract {
        kv_layout: BatchAttentionKvLayout::Linear,
        kv_format: BatchAttentionKvFormat::Fp16,
        kv_start: 0,
        ring_capacity: 0,
        sink_count: 0,
    };
    let plan = build_batch_attention_plan(
        input_contract,
        live_rows as usize,
        live_rows as usize,
        4095,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    .expect("maximum-tile batch plan");
    assert_eq!(plan.max_chunks, MAX_CHUNKS);

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let constants = attention_function_constants(scale, 0, plan.max_chunks, false);
    let constants_key = attention_constants_key(scale, 0, plan.max_chunks, false);
    let mut context = crate::MetalContext::new().expect("Metal device");
    let pipeline = context
        .pipeline(
            SOURCE,
            "attention_decode_batch_partial",
            &constants,
            &constants_key,
        )
        .expect("maximum-tile batch partial pipeline");

    let selected_threads = THREADS_PER_GROUP;
    let compiled_thread_limit = pipeline.max_total_threads_per_threadgroup();
    let static_threadgroup_bytes = pipeline.static_threadgroup_memory_length();
    // The batch partial kernel declares all threadgroup arrays statically and
    // does not request dynamic threadgroup memory from a compute encoder.
    let dynamic_threadgroup_bytes = 0u64;
    let device_threadgroup_limit = context.device().max_threadgroup_memory_length() as u64;
    let device_name = context.device().name();

    println!(
        "device={device_name:?} head_dim={head_dim} q_heads={num_q_heads} \
         kv_heads={num_kv_heads} live_rows={live_rows} max_chunks={} \
         selected_threads={selected_threads} compiled_thread_limit={compiled_thread_limit} \
         static_threadgroup_bytes={static_threadgroup_bytes} \
         dynamic_threadgroup_bytes={dynamic_threadgroup_bytes} \
         device_threadgroup_limit_bytes={device_threadgroup_limit}",
        plan.max_chunks,
    );

    assert!(
        compiled_thread_limit >= selected_threads,
        "the encoder selects {selected_threads} threads but the compiled pipeline permits only \
         {compiled_thread_limit}"
    );
    assert!(
        static_threadgroup_bytes + dynamic_threadgroup_bytes <= device_threadgroup_limit,
        "compiled threadgroup memory is {} bytes but device {device_name:?} supports only \
         {device_threadgroup_limit} bytes",
        static_threadgroup_bytes + dynamic_threadgroup_bytes,
    );
}

#[test]
fn batch_attention_plan_builds_contiguous_causal_rows_and_sizes_buffers() {
    let plan = build_linear_fp16_batch_attention_plan(3, 4, 31, 8, 2, 64).unwrap();

    assert_eq!(
        plan.row_plans,
        vec![
            RowChunkPlan {
                key_start: 0,
                seq_len: 32,
                chunk_len: 16,
                num_chunks: 2,
            },
            RowChunkPlan {
                key_start: 0,
                seq_len: 33,
                chunk_len: 17,
                num_chunks: 2,
            },
            RowChunkPlan {
                key_start: 0,
                seq_len: 34,
                chunk_len: 17,
                num_chunks: 2,
            },
        ]
    );
    assert_eq!(plan.live_rows, 3);
    assert_eq!(plan.capacity, 4);
    assert_eq!(plan.num_q_heads, 8);
    assert_eq!(plan.num_kv_heads, 2);
    assert_eq!(plan.head_dim, 64);
    assert_eq!(plan.max_chunks, 2);
    assert_eq!(plan.partial_threadgroups, 16);
    assert_eq!(plan.combine_threadgroups, 24);
    assert_eq!(
        plan.required_buffers,
        BatchAttentionBufferLengths {
            q_bytes: 3 * 8 * 64 * 2,
            k_bytes: 34 * 2 * 64 * 2,
            v_bytes: 34 * 2 * 64 * 2,
            output_bytes: 3 * 8 * 64 * 2,
            row_plan_bytes: 3 * 16,
            partial_bytes: 3 * 8 * 2 * (2 + 64) * 4,
        }
    );
}

#[test]
fn batch_attention_plan_accepts_position_zero_and_the_sixteen_chunk_cap() {
    let position_zero = build_linear_fp16_batch_attention_plan(1, 1, 0, 1, 1, 1).unwrap();
    assert_eq!(
        position_zero.row_plans,
        vec![RowChunkPlan {
            key_start: 0,
            seq_len: 1,
            chunk_len: 1,
            num_chunks: 1,
        }]
    );

    let max_chunks = build_linear_fp16_batch_attention_plan(1, 1, 255, 1, 1, 1).unwrap();
    assert_eq!(max_chunks.max_chunks, MAX_CHUNKS);
    assert_eq!(max_chunks.row_plans[0].chunk_len, 16);

    let max_head_dim = build_linear_fp16_batch_attention_plan(1, 1, 0, 1, 1, 512).unwrap();
    assert_eq!(max_head_dim.head_dim, 512);
}

#[test]
fn batch_attention_plan_rejects_invalid_rows_and_head_shapes() {
    for args in [
        (0, 1, 0, 1, 1, 1),
        (1, 0, 0, 1, 1, 1),
        (2, 1, 0, 1, 1, 1),
        (1, 1, 0, 0, 1, 1),
        (1, 1, 0, 1, 0, 1),
        (1, 1, 0, 3, 2, 1),
        (1, 1, 0, 1, 1, 0),
        (1, 1, 0, 1, 1, 513),
    ] {
        assert_invalid_input(build_linear_fp16_batch_attention_plan(
            args.0, args.1, args.2, args.3, args.4, args.5,
        ));
    }
}

#[test]
fn batch_attention_plan_rejects_position_and_byte_arithmetic_overflow() {
    assert_invalid_input(build_linear_fp16_batch_attention_plan(
        1,
        1,
        u32::MAX,
        1,
        1,
        1,
    ));

    let huge_rows = u32::MAX as usize;
    assert_invalid_input(build_linear_fp16_batch_attention_plan(
        huge_rows,
        huge_rows,
        0,
        u32::MAX,
        1,
        512,
    ));
}

#[test]
fn batch_attention_plan_rejects_malformed_internal_row_plans() {
    for malformed in [
        vec![],
        vec![RowChunkPlan {
            key_start: 1,
            seq_len: 32,
            chunk_len: 16,
            num_chunks: 2,
        }],
        vec![RowChunkPlan {
            key_start: 0,
            seq_len: 0,
            chunk_len: 1,
            num_chunks: 1,
        }],
        vec![RowChunkPlan {
            key_start: 0,
            seq_len: 32,
            chunk_len: 16,
            num_chunks: 1,
        }],
        vec![RowChunkPlan {
            key_start: 0,
            seq_len: 32,
            chunk_len: 15,
            num_chunks: 2,
        }],
        vec![
            RowChunkPlan {
                key_start: 0,
                seq_len: 32,
                chunk_len: 16,
                num_chunks: 2,
            },
            RowChunkPlan {
                key_start: 0,
                seq_len: 34,
                chunk_len: 17,
                num_chunks: 2,
            },
        ],
    ] {
        assert_invalid_input(validate_batch_row_plans(&malformed));
    }
}

#[test]
fn batch_attention_plan_rejects_each_undersized_buffer() {
    let plan = build_linear_fp16_batch_attention_plan(2, 4, 15, 4, 2, 32).unwrap();
    let exact = plan.required_buffers;
    plan.validate_buffer_lengths(exact).unwrap();

    for undersized in [
        BatchAttentionBufferLengths {
            q_bytes: exact.q_bytes - 1,
            ..exact
        },
        BatchAttentionBufferLengths {
            k_bytes: exact.k_bytes - 1,
            ..exact
        },
        BatchAttentionBufferLengths {
            v_bytes: exact.v_bytes - 1,
            ..exact
        },
        BatchAttentionBufferLengths {
            output_bytes: exact.output_bytes - 1,
            ..exact
        },
        BatchAttentionBufferLengths {
            row_plan_bytes: exact.row_plan_bytes - 1,
            ..exact
        },
        BatchAttentionBufferLengths {
            partial_bytes: exact.partial_bytes - 1,
            ..exact
        },
    ] {
        assert_invalid_input(plan.validate_buffer_lengths(undersized));
    }
}

fn assert_invalid_input<T>(result: Result<T, crate::GpuError>) {
    assert!(matches!(result, Err(crate::GpuError::InvalidInput(_))));
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

fn build_linear_fp16_batch_attention_plan(
    live_rows: usize,
    capacity: usize,
    first_query_position: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<super::BatchAttentionPlan, crate::GpuError> {
    build_batch_attention_plan(
        linear_fp16_contract(),
        live_rows,
        capacity,
        first_query_position,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
}

#[test]
fn batch_attention_plan_rejects_non_linear_cache_layout() {
    let contract = BatchAttentionInputContract {
        kv_layout: BatchAttentionKvLayout::NonLinear,
        ..linear_fp16_contract()
    };
    assert_invalid_input(build_batch_attention_plan(contract, 1, 1, 0, 1, 1, 1));
}

#[test]
fn batch_attention_plan_rejects_ring_cache_capacity() {
    let ring_backed = BatchAttentionInputContract {
        ring_capacity: 128,
        ..linear_fp16_contract()
    };
    assert_invalid_input(build_batch_attention_plan(ring_backed, 1, 1, 0, 1, 1, 1));
}

#[test]
fn batch_attention_plan_rejects_quantized_and_non_fp16_kv() {
    for kv_format in [
        BatchAttentionKvFormat::Quantized,
        BatchAttentionKvFormat::Other,
    ] {
        let contract = BatchAttentionInputContract {
            kv_format,
            ..linear_fp16_contract()
        };
        assert_invalid_input(build_batch_attention_plan(contract, 1, 1, 0, 1, 1, 1));
    }
}

#[test]
fn batch_attention_plan_rejects_nonzero_kv_start() {
    let nonzero_start = BatchAttentionInputContract {
        kv_start: 1,
        ..linear_fp16_contract()
    };
    assert_invalid_input(build_batch_attention_plan(nonzero_start, 1, 1, 0, 1, 1, 1));
}

#[test]
fn batch_attention_plan_rejects_attention_sink_count() {
    let with_sinks = BatchAttentionInputContract {
        sink_count: 1,
        ..linear_fp16_contract()
    };
    assert_invalid_input(build_batch_attention_plan(with_sinks, 1, 1, 0, 1, 1, 1));
}
