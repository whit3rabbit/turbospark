//! Safe Rust wrappers for the image-specific Metal kernels.
//!
//! These wrappers intentionally do not call the text runtime kernels. Image
//! activations and accumulators are FP32, while the packed image payload is a
//! per-row interleaved INT4 format. The only shared GPU contracts here are
//! `MetalContext`, `PassEncoder`, and the zero-copy resident buffer wrapper.

use std::{cell::RefCell, sync::Arc};

use gpu::{MetalContext, PassEncoder, ResidentGpuWeights};
use metal::{Buffer, ComputePipelineState, FunctionConstantValues};

const SOURCE: &str = include_str!("shaders/image.metal");
const THREADS: u64 = 256;

#[derive(Debug, Clone, Copy)]
pub(crate) struct WeightRef {
    pub(crate) offset: u64,
    pub(crate) storage: u32,
    pub(crate) row_stride: u32,
}

#[derive(Clone)]
pub(crate) struct GpuTensor {
    pub(crate) buffer: Buffer,
    pub(crate) len: usize,
    ready: Option<Arc<gpu::CommittedPass>>,
}

pub(crate) struct Component {
    pub(crate) store: crate::PackedTensorStore,
    pub(crate) resident: ResidentGpuWeights,
    latest_pass: RefCell<Option<Arc<gpu::CommittedPass>>>,
}

impl Component {
    pub(crate) fn open(context: &MetalContext, root: &std::path::Path) -> Result<Self, String> {
        let store = crate::PackedTensorStore::open(root)?;
        let length = std::fs::metadata(store.payload_path())
            .map_err(|e| {
                format!(
                    "failed to stat image payload {}: {e}",
                    store.payload_path().display()
                )
            })?
            .len();
        let mapped =
            model_io::ResidentBuffer::map(store.payload_path(), 0, length).map_err(|e| {
                format!(
                    "failed to map image payload {}: {e}",
                    store.payload_path().display()
                )
            })?;
        let resident = ResidentGpuWeights::wrap(context.device(), mapped)
            .map_err(|e| format!("failed to wrap image payload in Metal: {e}"))?;
        Ok(Self {
            store,
            resident,
            latest_pass: RefCell::new(None),
        })
    }

    pub(crate) fn weight(&self, name: &str, expected_shape: &[usize]) -> Result<WeightRef, String> {
        let tensor = self
            .store
            .tensor(name)
            .ok_or_else(|| format!("image tensor {name} is missing"))?;
        if tensor.shape != expected_shape {
            return Err(format!(
                "image tensor {name} has shape {:?}, expected {expected_shape:?}",
                tensor.shape
            ));
        }
        let storage = match tensor.storage_dtype.as_str() {
            "F32" => 1,
            "BF16" => 2,
            "INT4_AFFINE" => 3,
            other => {
                return Err(format!(
                    "image tensor {name} has unsupported storage {other}"
                ))
            }
        };
        let row_stride = if expected_shape.len() == 2 {
            let rows = u64::try_from(expected_shape[0])
                .map_err(|_| format!("image tensor {name} row count overflows"))?;
            u32::try_from(tensor.length / rows)
                .map_err(|_| format!("image tensor {name} row stride exceeds Metal parameters"))?
        } else {
            0
        };
        Ok(WeightRef {
            offset: self.resident.gpu_offset(tensor.offset),
            storage,
            row_stride,
        })
    }
}

impl Drop for Component {
    fn drop(&mut self) {
        // The resident Metal buffer aliases the mmap. Waiting for the latest
        // pass also completes every earlier pass on this component's queue.
        // This barrier is required on cancellation and error unwinding, where
        // no output tensor reaches the normal read barrier.
        if let Some(pass) = self.latest_pass.get_mut().take() {
            pass.wait_ref();
        }
    }
}

fn pipeline(
    context: &mut MetalContext,
    name: &'static str,
) -> Result<ComputePipelineState, String> {
    context
        .pipeline(SOURCE, name, &FunctionConstantValues::new(), b"")
        .map_err(|e| e.to_string())
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub(crate) fn upload_u32(context: &MetalContext, values: &[u32]) -> Buffer {
    context.new_buffer_with_data(values)
}

pub(crate) fn upload(context: &MetalContext, values: &[f32]) -> GpuTensor {
    GpuTensor {
        buffer: context.new_buffer_with_data(values),
        len: values.len(),
        ready: None,
    }
}

/// Materialize the FP32 buffer as BF16 values, retaining an FP32 buffer ABI.
///
/// This is diagnostic storage emulation for the Z-Image checkpoint probe. The
/// native image path remains FP32 unless the caller explicitly inserts this
/// pass between operations.
pub(crate) fn round_bf16(
    context: &mut MetalContext,
    input: &GpuTensor,
) -> Result<GpuTensor, String> {
    let output = GpuTensor {
        buffer: context.new_output_buffer((input.len * 4) as u64),
        len: input.len,
        ready: None,
    };
    let params = u32_bytes(&[input.len as u32]);
    let shader = pipeline(context, "image_round_bf16")?;
    let pass = context.begin_pass_labeled("image-round-bf16");
    dispatch(
        context,
        &pass,
        &shader,
        &[(&input.buffer, 0, 0), (&output.buffer, 1, 0)],
        &[(&params, 2)],
        output.len,
    );
    let ready = commit_deferred(pass);
    Ok(with_ready(output, ready))
}

pub(crate) fn read(tensor: &GpuTensor) -> Vec<f32> {
    if let Some(ready) = &tensor.ready {
        ready.wait_ref();
    }
    let bytes = gpu::read_buffer_bytes(&tensor.buffer, 0, tensor.len * 4);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// Image generation submits hundreds of small command buffers. Keep the
// readiness handle on each output so same-queue work can continue without a
// CPU wait, while every CPU read still waits for the producing pass. Record
// the latest pass on the component so its mmap cannot be released before an
// early-return teardown drains all resident-backed work.
fn commit_deferred(pass: PassEncoder) -> Arc<gpu::CommittedPass> {
    let ready = Arc::new(gpu::autorelease_pool(|| pass.commit()));
    // Diagnostic only: force a CPU completion after every primitive so a
    // seeded parity trace can distinguish same-queue hazard handling from
    // arithmetic or layout drift. The production path keeps the asynchronous
    // command-buffer chain unchanged.
    if std::env::var_os("TURBOSPARK_IMAGE_FORCE_SYNC_DISPATCH").is_some() {
        ready.wait_ref();
    }
    ready
}

fn commit_component_deferred(pass: PassEncoder, component: &Component) -> Arc<gpu::CommittedPass> {
    let ready = commit_deferred(pass);
    *component.latest_pass.borrow_mut() = Some(Arc::clone(&ready));
    ready
}

fn with_ready(mut output: GpuTensor, ready: Arc<gpu::CommittedPass>) -> GpuTensor {
    output.ready = Some(ready);
    output
}

fn dispatch(
    context: &MetalContext,
    pass: &PassEncoder,
    pipeline: &ComputePipelineState,
    buffers: &[(&Buffer, u64, u64)],
    bytes: &[(&[u8], u64)],
    count: usize,
) {
    pass.encode_threads_3d(
        pipeline,
        buffers,
        bytes,
        (count as u64, 1, 1),
        (THREADS.min(count.max(1) as u64), 1, 1),
    );
    let _ = context;
}

fn dispatch_tiled(
    pass: &PassEncoder,
    pipeline: &ComputePipelineState,
    buffers: &[(&Buffer, u64, u64)],
    bytes: &[(&[u8], u64)],
    rows: usize,
    columns: usize,
) {
    pass.encode_threadgroups_3d(
        pipeline,
        buffers,
        bytes,
        (columns.div_ceil(32) as u64, rows.div_ceil(8) as u64, 1),
        (32, 8, 1),
    );
}

pub(crate) fn lookup(
    context: &mut MetalContext,
    component: &Component,
    weight: WeightRef,
    ids: &[u32],
    vocab: usize,
    dim: usize,
) -> Result<GpuTensor, String> {
    let ids_buffer = upload_u32(context, ids);
    let output = GpuTensor {
        buffer: context.new_output_buffer((ids.len() * dim * 4) as u64),
        len: ids.len() * dim,
        ready: None,
    };
    let params = u32_bytes(&[ids.len() as u32, dim as u32, vocab as u32, weight.storage]);
    let shader = pipeline(context, "image_lookup")?;
    let pass = context.begin_pass_labeled("image-text-embedding");
    dispatch(
        context,
        &pass,
        &shader,
        &[
            (component.resident.buffer(), 0, weight.offset),
            (&ids_buffer, 1, 0),
            (&output.buffer, 2, 0),
        ],
        &[(&params, 3)],
        output.len,
    );
    let ready = commit_component_deferred(pass, component);
    Ok(with_ready(output, ready))
}

// Keep the wrapper arguments flat: each one is part of the image kernel's
// explicit shape and storage contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear(
    context: &mut MetalContext,
    component: &Component,
    weight: WeightRef,
    bias: Option<WeightRef>,
    input: &GpuTensor,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<GpuTensor, String> {
    if input.len != rows * in_dim {
        return Err("image linear input shape does not match rows and in_dim".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((rows * out_dim * 4) as u64),
        len: rows * out_dim,
        ready: None,
    };
    let params = u32_bytes(&[
        rows as u32,
        in_dim as u32,
        out_dim as u32,
        weight.row_stride,
        weight.storage,
        bias.map_or(0, |value| value.storage),
    ]);
    let shader = pipeline(context, "image_linear_tiled")?;
    let pass = context.begin_pass_labeled("image-linear");
    let mut buffers = vec![
        (component.resident.buffer(), 0, weight.offset),
        (&input.buffer, 1, 0),
        (&output.buffer, 2, 0),
    ];
    if let Some(bias) = bias {
        buffers.push((component.resident.buffer(), 3, bias.offset));
    }
    dispatch_tiled(&pass, &shader, &buffers, &[(&params, 4)], rows, out_dim);
    let ready = commit_component_deferred(pass, component);
    Ok(with_ready(output, ready))
}

pub(crate) fn rms_norm(
    context: &mut MetalContext,
    component: &Component,
    weight: WeightRef,
    input: &GpuTensor,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<GpuTensor, String> {
    if input.len != rows * dim {
        return Err("image RMSNorm input shape does not match rows and dim".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((input.len * 4) as u64),
        len: input.len,
        ready: None,
    };
    let mut params = u32_bytes(&[rows as u32, dim as u32, weight.storage]);
    params.extend_from_slice(&eps.to_le_bytes());
    let shader = pipeline(context, "image_rms_norm")?;
    let pass = context.begin_pass_labeled("image-rms-norm");
    pass.encode_threadgroups_3d(
        &shader,
        &[
            (&input.buffer, 0, 0),
            (component.resident.buffer(), 1, weight.offset),
            (&output.buffer, 2, 0),
        ],
        &[(&params, 3)],
        (rows as u64, 1, 1),
        (THREADS, 1, 1),
    );
    let ready = commit_component_deferred(pass, component);
    Ok(with_ready(output, ready))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rope(
    context: &mut MetalContext,
    component: &Component,
    weight: WeightRef,
    input: &GpuTensor,
    freqs: &Buffer,
    rows: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<GpuTensor, String> {
    if input.len != rows * heads * head_dim {
        return Err("image RoPE input shape does not match rows, heads, and head_dim".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((input.len * 4) as u64),
        len: input.len,
        ready: None,
    };
    let mut params = u32_bytes(&[rows as u32, heads as u32, head_dim as u32, weight.storage]);
    params.extend_from_slice(&eps.to_le_bytes());
    let shader = pipeline(context, "image_rope_orthogonal")?;
    let pass = context.begin_pass_labeled("image-rope");
    dispatch(
        context,
        &pass,
        &shader,
        &[
            (&input.buffer, 0, 0),
            (&output.buffer, 1, 0),
            (component.resident.buffer(), 2, weight.offset),
            (freqs, 3, 0),
        ],
        &[(&params, 4)],
        output.len,
    );
    let ready = commit_component_deferred(pass, component);
    Ok(with_ready(output, ready))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn adjacent_rope(
    context: &mut MetalContext,
    component: &Component,
    weight: WeightRef,
    input: &GpuTensor,
    freqs: &Buffer,
    rows: usize,
    heads: usize,
    dim: usize,
    eps: f32,
) -> Result<GpuTensor, String> {
    if input.len != rows * heads * dim {
        return Err("image adjacent RoPE input shape does not match".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((input.len * 4) as u64),
        len: input.len,
        ready: None,
    };
    let mut params = u32_bytes(&[rows as u32, heads as u32, dim as u32, weight.storage]);
    params.extend_from_slice(&eps.to_le_bytes());
    let shader = pipeline(context, "image_rope_adjacent")?;
    let pass = context.begin_pass_labeled("image-adjacent-rope");
    pass.encode_threadgroups_3d(
        &shader,
        &[
            (&input.buffer, 0, 0),
            (&output.buffer, 1, 0),
            (component.resident.buffer(), 2, weight.offset),
            (freqs, 3, 0),
        ],
        &[(&params, 4)],
        ((rows * heads) as u64, 1, 1),
        (THREADS, 1, 1),
    );
    let ready = commit_component_deferred(pass, component);
    Ok(with_ready(output, ready))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    context: &mut MetalContext,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    q_rows: usize,
    kv_rows: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<GpuTensor, String> {
    if q_rows == 0 || kv_rows == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 {
        return Err("image attention dimensions must be non-zero".to_string());
    }
    if q_heads % kv_heads != 0 || head_dim > 128 || head_dim % 32 != 0 {
        return Err("image attention dimensions are not supported by the Metal kernel".to_string());
    }
    let expected_q = q_rows
        .checked_mul(q_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "image attention query shape overflowed".to_string())?;
    let expected_kv = kv_rows
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "image attention key/value shape overflowed".to_string())?;
    let threadgroups = q_heads
        .checked_mul(q_rows.div_ceil(4))
        .ok_or_else(|| "image attention grid shape overflowed".to_string())?;
    if q.len != expected_q || k.len != expected_kv || v.len != expected_kv {
        return Err("image attention tensor lengths do not match the requested shape".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((q.len * 4) as u64),
        len: q.len,
        ready: None,
    };
    let params = u32_bytes(&[
        q_rows as u32,
        kv_rows as u32,
        q_heads as u32,
        kv_heads as u32,
        head_dim as u32,
        u32::from(causal),
    ]);
    let shader = pipeline(context, "image_attention")?;
    let pass = context.begin_pass_labeled("image-attention");
    pass.encode_threadgroups_3d(
        &shader,
        &[
            (&q.buffer, 0, 0),
            (&k.buffer, 1, 0),
            (&v.buffer, 2, 0),
            (&output.buffer, 3, 0),
        ],
        &[(&params, 4)],
        (threadgroups as u64, 1, 1),
        ((head_dim * 4) as u64, 1, 1),
    );
    let ready = commit_deferred(pass);
    Ok(with_ready(output, ready))
}

pub(crate) fn add(
    context: &mut MetalContext,
    left: &GpuTensor,
    right: &GpuTensor,
) -> Result<GpuTensor, String> {
    if left.len != right.len {
        return Err("image add operands have different lengths".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((left.len * 4) as u64),
        len: left.len,
        ready: None,
    };
    let params = u32_bytes(&[left.len as u32]);
    let shader = pipeline(context, "image_add")?;
    let pass = context.begin_pass_labeled("image-add");
    dispatch(
        context,
        &pass,
        &shader,
        &[
            (&left.buffer, 0, 0),
            (&right.buffer, 1, 0),
            (&output.buffer, 2, 0),
        ],
        &[(&params, 3)],
        output.len,
    );
    let ready = commit_deferred(pass);
    Ok(with_ready(output, ready))
}

pub(crate) fn silu_mul(
    context: &mut MetalContext,
    left: &GpuTensor,
    right: &GpuTensor,
) -> Result<GpuTensor, String> {
    if left.len != right.len {
        return Err("image SiLU operands have different lengths".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((left.len * 4) as u64),
        len: left.len,
        ready: None,
    };
    let params = u32_bytes(&[left.len as u32]);
    let shader = pipeline(context, "image_silu_mul")?;
    let pass = context.begin_pass_labeled("image-silu-mul");
    dispatch(
        context,
        &pass,
        &shader,
        &[
            (&left.buffer, 0, 0),
            (&right.buffer, 1, 0),
            (&output.buffer, 2, 0),
        ],
        &[(&params, 3)],
        output.len,
    );
    let ready = commit_deferred(pass);
    Ok(with_ready(output, ready))
}

pub(crate) fn scale_rows(
    context: &mut MetalContext,
    input: &GpuTensor,
    scale: &GpuTensor,
    rows: usize,
    dim: usize,
) -> Result<GpuTensor, String> {
    if input.len != rows * dim || scale.len != dim {
        return Err("image scale input shape does not match rows and dim".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((input.len * 4) as u64),
        len: input.len,
        ready: None,
    };
    let params = u32_bytes(&[rows as u32, dim as u32]);
    let shader = pipeline(context, "image_scale_shift")?;
    let pass = context.begin_pass_labeled("image-scale");
    dispatch(
        context,
        &pass,
        &shader,
        &[
            (&input.buffer, 0, 0),
            (&scale.buffer, 1, 0),
            (&output.buffer, 2, 0),
        ],
        &[(&params, 3)],
        output.len,
    );
    let ready = commit_deferred(pass);
    Ok(with_ready(output, ready))
}

pub(crate) fn gate_add(
    context: &mut MetalContext,
    residual: &GpuTensor,
    value: &GpuTensor,
    gate: &GpuTensor,
    rows: usize,
    dim: usize,
) -> Result<GpuTensor, String> {
    if residual.len != rows * dim || value.len != residual.len || gate.len != dim {
        return Err("image gated residual shapes do not match".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((residual.len * 4) as u64),
        len: residual.len,
        ready: None,
    };
    let params = u32_bytes(&[rows as u32, dim as u32]);
    let shader = pipeline(context, "image_gate_add")?;
    let pass = context.begin_pass_labeled("image-gated-residual");
    dispatch(
        context,
        &pass,
        &shader,
        &[
            (&residual.buffer, 0, 0),
            (&value.buffer, 1, 0),
            (&gate.buffer, 2, 0),
            (&output.buffer, 3, 0),
        ],
        &[(&params, 4)],
        output.len,
    );
    let ready = commit_deferred(pass);
    Ok(with_ready(output, ready))
}

pub(crate) fn layer_norm(
    context: &mut MetalContext,
    input: &GpuTensor,
    scale: &GpuTensor,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<GpuTensor, String> {
    if input.len != rows * dim || scale.len != dim {
        return Err("image LayerNorm shapes do not match".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((input.len * 4) as u64),
        len: input.len,
        ready: None,
    };
    let mut params = u32_bytes(&[rows as u32, dim as u32, 1]);
    params.extend_from_slice(&eps.to_le_bytes());
    let shader = pipeline(context, "image_layer_norm")?;
    let pass = context.begin_pass_labeled("image-layer-norm");
    dispatch(
        context,
        &pass,
        &shader,
        &[
            (&input.buffer, 0, 0),
            (&scale.buffer, 1, 0),
            (&output.buffer, 2, 0),
        ],
        &[(&params, 3)],
        output.len,
    );
    let ready = commit_deferred(pass);
    Ok(with_ready(output, ready))
}

pub(crate) fn scheduler_step(
    context: &mut MetalContext,
    sample: &GpuTensor,
    velocity: &GpuTensor,
    delta: f32,
) -> Result<GpuTensor, String> {
    if sample.len != velocity.len {
        return Err("image scheduler operands have different lengths".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((sample.len * 4) as u64),
        len: sample.len,
        ready: None,
    };
    let params = delta.to_le_bytes();
    let count = u32_bytes(&[sample.len as u32]);
    let shader = pipeline(context, "image_scheduler_step")?;
    let pass = context.begin_pass_labeled("image-scheduler");
    dispatch(
        context,
        &pass,
        &shader,
        &[
            (&sample.buffer, 0, 0),
            (&velocity.buffer, 1, 0),
            (&output.buffer, 2, 0),
        ],
        &[(&params, 3), (&count, 4)],
        output.len,
    );
    let ready = commit_deferred(pass);
    Ok(with_ready(output, ready))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d(
    context: &mut MetalContext,
    component: &Component,
    weight: WeightRef,
    bias: Option<WeightRef>,
    input: &GpuTensor,
    channels: usize,
    height: usize,
    width: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<GpuTensor, String> {
    if input.len != channels * height * width {
        return Err("image convolution input shape does not match".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((out_channels * height * width * 4) as u64),
        len: out_channels * height * width,
        ready: None,
    };
    let params = u32_bytes(&[
        channels as u32,
        out_channels as u32,
        height as u32,
        width as u32,
        kernel as u32,
        stride as u32,
        padding as u32,
        weight.storage,
        bias.map_or(0, |value| value.storage),
    ]);
    let shader = pipeline(context, "image_conv2d")?;
    let pass = context.begin_pass_labeled("image-conv2d");
    let mut buffers = vec![
        (component.resident.buffer(), 0, weight.offset),
        (&input.buffer, 2, 0),
        (&output.buffer, 3, 0),
    ];
    if let Some(bias) = bias {
        buffers.push((component.resident.buffer(), 1, bias.offset));
    }
    pass.encode_threadgroups_3d(
        &shader,
        &buffers,
        &[(&params, 4)],
        (
            out_channels.div_ceil(8) as u64,
            (height * width).div_ceil(8) as u64,
            1,
        ),
        (256, 1, 1),
    );
    let ready = commit_component_deferred(pass, component);
    Ok(with_ready(output, ready))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn group_norm(
    context: &mut MetalContext,
    component: &Component,
    weight: WeightRef,
    bias: WeightRef,
    input: &GpuTensor,
    channels: usize,
    height: usize,
    width: usize,
    groups: usize,
    eps: f32,
) -> Result<GpuTensor, String> {
    if input.len != channels * height * width {
        return Err("image GroupNorm input shape does not match".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((input.len * 4) as u64),
        len: input.len,
        ready: None,
    };
    let mut params = u32_bytes(&[
        channels as u32,
        height as u32,
        width as u32,
        groups as u32,
        weight.storage,
        bias.storage,
    ]);
    params.extend_from_slice(&eps.to_le_bytes());
    let shader = pipeline(context, "image_group_norm")?;
    let pass = context.begin_pass_labeled("image-group-norm");
    pass.encode_threadgroups_3d(
        &shader,
        &[
            (&input.buffer, 0, 0),
            (component.resident.buffer(), 1, weight.offset),
            (component.resident.buffer(), 2, bias.offset),
            (&output.buffer, 3, 0),
        ],
        &[(&params, 4)],
        (groups as u64, 1, 1),
        (256, 1, 1),
    );
    let ready = commit_component_deferred(pass, component);
    Ok(with_ready(output, ready))
}

pub(crate) fn upsample(
    context: &mut MetalContext,
    component: &Component,
    input: &GpuTensor,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<GpuTensor, String> {
    if input.len != channels * height * width {
        return Err("image upsample input shape does not match".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((channels * height * 2 * width * 2 * 4) as u64),
        len: channels * height * 2 * width * 2,
        ready: None,
    };
    let params = u32_bytes(&[channels as u32, height as u32, width as u32]);
    let shader = pipeline(context, "image_upsample_nearest")?;
    let pass = context.begin_pass_labeled("image-upsample");
    dispatch(
        context,
        &pass,
        &shader,
        &[(&input.buffer, 0, 0), (&output.buffer, 1, 0)],
        &[(&params, 2)],
        output.len,
    );
    let ready = commit_component_deferred(pass, component);
    Ok(with_ready(output, ready))
}

pub(crate) fn silu(
    context: &mut MetalContext,
    component: &Component,
    input: &GpuTensor,
) -> Result<GpuTensor, String> {
    let output = GpuTensor {
        buffer: context.new_output_buffer((input.len * 4) as u64),
        len: input.len,
        ready: None,
    };
    let params = u32_bytes(&[input.len as u32]);
    let shader = pipeline(context, "image_silu")?;
    let pass = context.begin_pass_labeled("image-silu");
    dispatch(
        context,
        &pass,
        &shader,
        &[(&input.buffer, 0, 0), (&output.buffer, 1, 0)],
        &[(&params, 2)],
        output.len,
    );
    let ready = commit_component_deferred(pass, component);
    Ok(with_ready(output, ready))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vae_attention(
    context: &mut MetalContext,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<GpuTensor, String> {
    if channels == 0 || channels > 512 {
        return Err("image VAE attention channels must be in 1..=512".to_string());
    }
    let len = channels * height * width;
    if q.len != len || k.len != len || v.len != len {
        return Err("image VAE attention input shapes do not match".to_string());
    }
    let output = GpuTensor {
        buffer: context.new_output_buffer((len * 4) as u64),
        len,
        ready: None,
    };
    let params = u32_bytes(&[channels as u32, height as u32, width as u32]);
    let shader = pipeline(context, "image_vae_attention")?;
    let pass = context.begin_pass_labeled("image-vae-attention");
    pass.encode_threadgroups_3d(
        &shader,
        &[
            (&q.buffer, 0, 0),
            (&k.buffer, 1, 0),
            (&v.buffer, 2, 0),
            (&output.buffer, 3, 0),
        ],
        &[(&params, 4)],
        ((len / channels).div_ceil(8) as u64, 1, 1),
        (256, 1, 1),
    );
    let ready = commit_deferred(pass);
    Ok(with_ready(output, ready))
}

pub(crate) fn frequencies(rows: usize, dim: usize, theta: f32) -> Vec<f32> {
    let half = dim / 2;
    let log_theta = theta.ln();
    (0..rows)
        .flat_map(|row| {
            (0..half)
                .map(move |pair| row as f32 * (-((2 * pair) as f32) / dim as f32 * log_theta).exp())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::{
        compute_tensor_inventory_sha256, PackedIndex, PackedQuantization, PackedTensor,
        PACKED_DATA_NAME, PACKED_GROUP_SIZE, PACKED_MAGIC, PACKED_VERSION,
    };
    use compute::quantize_int4_affine;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    #[ignore = "opt-in Metal kernel parity test"]
    fn tiled_linear_matches_f32_and_interleaved_int4_packed_rows() {
        let Ok(mut context) = MetalContext::new() else {
            eprintln!("NOTE: skipping image Metal parity test, no Metal device");
            return;
        };
        let root = temporary_directory();
        let rows = 9;
        let in_dim = 64;
        let out_dim = 10;
        let matrix: Vec<f32> = (0..out_dim * in_dim)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 37.0)
            .collect();
        let input: Vec<f32> = (0..rows * in_dim)
            .map(|index| ((index * 11 % 47) as f32 - 23.0) / 19.0)
            .collect();
        let bias: Vec<f32> = (0..out_dim)
            .map(|index| (index as f32 - 4.0) / 13.0)
            .collect();
        let int4_matrix: Vec<f32> = matrix.iter().map(|value| value * 0.7 + 0.13).collect();

        let mut payload = Vec::new();
        let matrix_offset = append_f32(&mut payload, &matrix);
        let bias_offset = append_f32(&mut payload, &bias);
        let int4_offset = payload.len() as u64;
        let int4_row_bytes = in_dim / 2 + (in_dim / PACKED_GROUP_SIZE) * 4;
        for row in int4_matrix.chunks_exact(in_dim) {
            let packed = quantize_int4_affine(row);
            payload.extend_from_slice(&packed.packed);
            for value in packed.scales.iter().chain(&packed.biases) {
                payload.extend_from_slice(&value.to_le_bytes());
            }
        }
        assert_eq!(
            payload.len() as u64 - int4_offset,
            (out_dim * int4_row_bytes) as u64
        );

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "matrix.weight".to_string(),
            PackedTensor {
                shape: vec![out_dim, in_dim],
                source_dtype: "F32".to_string(),
                storage_dtype: "F32".to_string(),
                offset: matrix_offset,
                length: (matrix.len() * 4) as u64,
                quantization: None,
            },
        );
        tensors.insert(
            "matrix.bias".to_string(),
            PackedTensor {
                shape: vec![out_dim],
                source_dtype: "F32".to_string(),
                storage_dtype: "F32".to_string(),
                offset: bias_offset,
                length: (bias.len() * 4) as u64,
                quantization: None,
            },
        );
        tensors.insert(
            "matrix.int4".to_string(),
            PackedTensor {
                shape: vec![out_dim, in_dim],
                source_dtype: "F32".to_string(),
                storage_dtype: "INT4_AFFINE".to_string(),
                offset: int4_offset,
                length: (out_dim * int4_row_bytes) as u64,
                quantization: Some(PackedQuantization {
                    scheme: "four-bit-linear-weights-group-64".to_string(),
                    group_size: PACKED_GROUP_SIZE,
                    nibble_order: "low_nibble_even_high_nibble_odd".to_string(),
                    scale_and_bias_convention: "value = nibble * bf16_scale + bf16_bias"
                        .to_string(),
                }),
            },
        );
        write_index(&root, payload, tensors);

        let component = Component::open(&context, &root).expect("open synthetic component");
        let input_gpu = upload(&context, &input);
        let f32_output = linear(
            &mut context,
            &component,
            component
                .weight("matrix.weight", &[out_dim, in_dim])
                .unwrap(),
            Some(component.weight("matrix.bias", &[out_dim]).unwrap()),
            &input_gpu,
            rows,
            in_dim,
            out_dim,
        )
        .expect("F32 tiled linear");
        let expected_f32 = cpu_linear(&input, &matrix, &bias, rows, in_dim, out_dim);
        assert_close(&read(&f32_output), &expected_f32, 2e-5);

        let int4_output = linear(
            &mut context,
            &component,
            component.weight("matrix.int4", &[out_dim, in_dim]).unwrap(),
            None,
            &input_gpu,
            rows,
            in_dim,
            out_dim,
        )
        .expect("INT4 tiled linear");
        let dequantized: Vec<f32> = int4_matrix
            .chunks_exact(in_dim)
            .flat_map(|row| {
                let packed = quantize_int4_affine(row);
                compute::dequantize_int4_affine(&packed, in_dim)
            })
            .collect();
        let expected_int4 = cpu_linear(&input, &dequantized, &[], rows, in_dim, out_dim);
        assert_close(&read(&int4_output), &expected_int4, 2e-5);
        drop(component);
        fs::remove_dir_all(root).expect("remove synthetic component");
    }

    #[test]
    #[ignore = "opt-in Metal kernel parity test"]
    fn simd_conv2d_matches_cpu_for_padded_stride_one() {
        let Ok(mut context) = MetalContext::new() else {
            eprintln!("NOTE: skipping image Metal parity test, no Metal device");
            return;
        };
        let channels = 5;
        let out_channels = 9;
        let height = 4;
        let width = 3;
        let kernel = 3;
        let matrix: Vec<f32> = (0..out_channels * channels * kernel * kernel)
            .map(|index| ((index * 11 % 71) as f32 - 35.0) / 43.0)
            .collect();
        let bias: Vec<f32> = (0..out_channels)
            .map(|index| (index as f32 - 4.0) / 17.0)
            .collect();
        let input: Vec<f32> = (0..channels * height * width)
            .map(|index| ((index * 7 % 53) as f32 - 26.0) / 31.0)
            .collect();
        let mut payload = Vec::new();
        let weight_offset = append_f32(&mut payload, &matrix);
        let bias_offset = append_f32(&mut payload, &bias);
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "conv.weight".to_string(),
            PackedTensor {
                shape: vec![out_channels, channels, kernel, kernel],
                source_dtype: "F32".to_string(),
                storage_dtype: "F32".to_string(),
                offset: weight_offset,
                length: (matrix.len() * 4) as u64,
                quantization: None,
            },
        );
        tensors.insert(
            "conv.bias".to_string(),
            PackedTensor {
                shape: vec![out_channels],
                source_dtype: "F32".to_string(),
                storage_dtype: "F32".to_string(),
                offset: bias_offset,
                length: (bias.len() * 4) as u64,
                quantization: None,
            },
        );
        let root = temporary_directory();
        write_index(&root, payload, tensors);
        let component = Component::open(&context, &root).expect("open synthetic component");
        let input_gpu = upload(&context, &input);
        let actual = conv2d(
            &mut context,
            &component,
            component
                .weight("conv.weight", &[out_channels, channels, kernel, kernel])
                .unwrap(),
            Some(component.weight("conv.bias", &[out_channels]).unwrap()),
            &input_gpu,
            channels,
            height,
            width,
            out_channels,
            kernel,
            1,
            1,
        )
        .expect("SIMD convolution");

        let mut expected = vec![0.0f32; out_channels * height * width];
        for oc in 0..out_channels {
            for oy in 0..height {
                for ox in 0..width {
                    let mut sum = bias[oc];
                    for ic in 0..channels {
                        for ky in 0..kernel {
                            for kx in 0..kernel {
                                let iy = oy as isize + ky as isize - 1;
                                let ix = ox as isize + kx as isize - 1;
                                if iy >= 0 && ix >= 0 && iy < height as isize && ix < width as isize
                                {
                                    let input_index =
                                        (ic * height + iy as usize) * width + ix as usize;
                                    let weight_index =
                                        ((oc * channels + ic) * kernel + ky) * kernel + kx;
                                    sum += input[input_index] * matrix[weight_index];
                                }
                            }
                        }
                    }
                    expected[(oc * height + oy) * width + ox] = sum;
                }
            }
        }
        assert_close(&read(&actual), &expected, 2e-5);
        drop(component);
        fs::remove_dir_all(root).expect("remove synthetic component");
    }

    #[test]
    #[ignore = "opt-in Metal kernel parity test"]
    fn cooperative_group_norm_matches_cpu() {
        let Ok(mut context) = MetalContext::new() else {
            eprintln!("NOTE: skipping image Metal parity test, no Metal device");
            return;
        };
        let channels = 8;
        let groups = 2;
        let height = 3;
        let width = 5;
        let eps = 1e-5;
        let input: Vec<f32> = (0..channels * height * width)
            .map(|index| ((index * 17 % 61) as f32 - 30.0) / 29.0)
            .collect();
        let weight: Vec<f32> = (0..channels)
            .map(|index| 0.7 + index as f32 / 23.0)
            .collect();
        let bias: Vec<f32> = (0..channels)
            .map(|index| (index as f32 - 3.0) / 19.0)
            .collect();
        let mut payload = Vec::new();
        let weight_offset = append_f32(&mut payload, &weight);
        let bias_offset = append_f32(&mut payload, &bias);
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "norm.weight".to_string(),
            PackedTensor {
                shape: vec![channels],
                source_dtype: "F32".to_string(),
                storage_dtype: "F32".to_string(),
                offset: weight_offset,
                length: (weight.len() * 4) as u64,
                quantization: None,
            },
        );
        tensors.insert(
            "norm.bias".to_string(),
            PackedTensor {
                shape: vec![channels],
                source_dtype: "F32".to_string(),
                storage_dtype: "F32".to_string(),
                offset: bias_offset,
                length: (bias.len() * 4) as u64,
                quantization: None,
            },
        );
        let root = temporary_directory();
        write_index(&root, payload, tensors);
        let component = Component::open(&context, &root).expect("open synthetic component");
        let input_gpu = upload(&context, &input);
        let actual = group_norm(
            &mut context,
            &component,
            component.weight("norm.weight", &[channels]).unwrap(),
            component.weight("norm.bias", &[channels]).unwrap(),
            &input_gpu,
            channels,
            height,
            width,
            groups,
            eps,
        )
        .expect("cooperative GroupNorm");
        let expected =
            crate::vae::group_norm(&input, channels, height, width, &weight, &bias, groups, eps)
                .expect("CPU GroupNorm");
        assert_close(&read(&actual), &expected, 2e-5);
        drop(component);
        fs::remove_dir_all(root).expect("remove synthetic component");
    }

    #[test]
    #[ignore = "opt-in Metal kernel parity test"]
    fn grouped_attention_matches_cpu_for_gqa_and_causal_mask() {
        let Ok(mut context) = MetalContext::new() else {
            eprintln!("NOTE: skipping image Metal parity test, no Metal device");
            return;
        };
        let q_rows = 5;
        let kv_rows = 6;
        let q_heads = 2;
        let kv_heads = 1;
        let head_dim = 32;
        let q: Vec<f32> = (0..q_rows * q_heads * head_dim)
            .map(|index| ((index * 13 % 71) as f32 - 35.0) / 53.0)
            .collect();
        let k: Vec<f32> = (0..kv_rows * kv_heads * head_dim)
            .map(|index| ((index * 7 % 61) as f32 - 30.0) / 47.0)
            .collect();
        let v: Vec<f32> = (0..kv_rows * kv_heads * head_dim)
            .map(|index| ((index * 19 % 83) as f32 - 41.0) / 67.0)
            .collect();
        let q_gpu = upload(&context, &q);
        let k_gpu = upload(&context, &k);
        let v_gpu = upload(&context, &v);

        for causal in [false, true] {
            let actual = attention(
                &mut context,
                &q_gpu,
                &k_gpu,
                &v_gpu,
                q_rows,
                kv_rows,
                q_heads,
                kv_heads,
                head_dim,
                causal,
            )
            .expect("grouped attention");
            let expected = cpu_attention(
                &q, &k, &v, q_rows, kv_rows, q_heads, kv_heads, head_dim, causal,
            );
            assert_close(&read(&actual), &expected, 2e-5);
        }

        // The production DiT uses 30 heads of width 128. Keep a small exact
        // shape case here because the grouped kernel's SIMD and query-slot
        // mapping changes at the four-SIMD-group head width.
        let q_rows = 8;
        let kv_rows = 8;
        let q_heads = 30;
        let kv_heads = 30;
        let head_dim = 128;
        let q: Vec<f32> = (0..q_rows * q_heads * head_dim)
            .map(|index| ((index * 13 % 71) as f32 - 35.0) / 53.0)
            .collect();
        let k: Vec<f32> = (0..kv_rows * kv_heads * head_dim)
            .map(|index| ((index * 7 % 61) as f32 - 30.0) / 47.0)
            .collect();
        let v: Vec<f32> = (0..kv_rows * kv_heads * head_dim)
            .map(|index| ((index * 19 % 83) as f32 - 41.0) / 67.0)
            .collect();
        let q_gpu = upload(&context, &q);
        let k_gpu = upload(&context, &k);
        let v_gpu = upload(&context, &v);
        let actual = attention(
            &mut context,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            q_rows,
            kv_rows,
            q_heads,
            kv_heads,
            head_dim,
            false,
        )
        .expect("production-shape grouped attention");
        let expected = cpu_attention(
            &q, &k, &v, q_rows, kv_rows, q_heads, kv_heads, head_dim, false,
        );
        assert_close(&read(&actual), &expected, 2e-5);
    }

    #[test]
    #[ignore = "opt-in Metal kernel parity test"]
    fn vae_attention_matches_cpu_for_production_channels() {
        let Ok(mut context) = MetalContext::new() else {
            eprintln!("NOTE: skipping image Metal parity test, no Metal device");
            return;
        };
        let channels = 512;
        let height = 2;
        let width = 4;
        let area = height * width;
        let len = channels * area;
        let q: Vec<f32> = (0..len)
            .map(|index| ((index * 13 % 97) as f32 - 48.0) / 97.0)
            .collect();
        let k: Vec<f32> = (0..len)
            .map(|index| ((index * 17 % 89) as f32 - 44.0) / 89.0)
            .collect();
        let v: Vec<f32> = (0..len)
            .map(|index| ((index * 19 % 83) as f32 - 41.0) / 83.0)
            .collect();
        let q_gpu = upload(&context, &q);
        let k_gpu = upload(&context, &k);
        let v_gpu = upload(&context, &v);
        let actual = vae_attention(
            &mut context,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            channels,
            height,
            width,
        )
        .expect("VAE attention");

        let scale = (channels as f32).sqrt().recip();
        let mut expected = vec![0.0f32; len];
        for channel in 0..channels {
            for query in 0..area {
                let mut maximum = f32::NEG_INFINITY;
                for key in 0..area {
                    let mut score = 0.0f32;
                    for c in 0..channels {
                        score += q[c * area + query] * k[c * area + key];
                    }
                    maximum = maximum.max(score * scale);
                }
                let mut denominator = 0.0f32;
                for key in 0..area {
                    let mut score = 0.0f32;
                    for c in 0..channels {
                        score += q[c * area + query] * k[c * area + key];
                    }
                    let probability = (score * scale - maximum).exp();
                    denominator += probability;
                    expected[channel * area + query] += probability * v[channel * area + key];
                }
                expected[channel * area + query] /= denominator;
            }
        }
        assert_close(&read(&actual), &expected, 2e-5);
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_rows: usize,
        kv_rows: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Vec<f32> {
        let mut output = vec![0.0; q.len()];
        let group_size = q_heads / kv_heads;
        let scale = (head_dim as f32).sqrt().recip();
        for query in 0..q_rows {
            for head in 0..q_heads {
                let kv_head = head / group_size;
                let mut maximum = f32::NEG_INFINITY;
                for key in 0..kv_rows {
                    if causal && key > query {
                        continue;
                    }
                    let score = (0..head_dim)
                        .map(|d| {
                            q[(query * q_heads + head) * head_dim + d]
                                * k[(key * kv_heads + kv_head) * head_dim + d]
                        })
                        .sum::<f32>()
                        * scale;
                    maximum = maximum.max(score);
                }
                let mut denominator = 0.0;
                for key in 0..kv_rows {
                    if causal && key > query {
                        continue;
                    }
                    let score = (0..head_dim)
                        .map(|d| {
                            q[(query * q_heads + head) * head_dim + d]
                                * k[(key * kv_heads + kv_head) * head_dim + d]
                        })
                        .sum::<f32>()
                        * scale;
                    let probability = (score - maximum).exp();
                    denominator += probability;
                    for d in 0..head_dim {
                        output[(query * q_heads + head) * head_dim + d] +=
                            probability * v[(key * kv_heads + kv_head) * head_dim + d];
                    }
                }
                for d in 0..head_dim {
                    output[(query * q_heads + head) * head_dim + d] /= denominator;
                }
            }
        }
        output
    }

    fn append_f32(payload: &mut Vec<u8>, values: &[f32]) -> u64 {
        let offset = payload.len() as u64;
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        offset
    }

    fn cpu_linear(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        (0..rows)
            .flat_map(|row| {
                (0..out_dim).map(move |column| {
                    let sum = (0..in_dim)
                        .map(|index| input[row * in_dim + index] * weight[column * in_dim + index])
                        .sum::<f32>();
                    if bias.is_empty() {
                        sum
                    } else {
                        sum + bias[column]
                    }
                })
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= tolerance,
            "max error {max_error} > {tolerance}"
        );
    }

    fn write_index(
        root: &std::path::Path,
        payload: Vec<u8>,
        tensors: BTreeMap<String, PackedTensor>,
    ) {
        fs::create_dir_all(root).expect("create synthetic component");
        fs::write(root.join(PACKED_DATA_NAME), &payload).expect("write synthetic payload");
        let index = PackedIndex {
            magic: PACKED_MAGIC.to_string(),
            version: PACKED_VERSION,
            data_file: PACKED_DATA_NAME.to_string(),
            data_sha256: model_io::hash_data(&payload),
            tensor_inventory_sha256: compute_tensor_inventory_sha256(&tensors),
            tensors,
        };
        fs::write(
            root.join(crate::packed::PACKED_INDEX_NAME),
            serde_json::to_vec(&index).expect("serialize synthetic index"),
        )
        .expect("write synthetic index");
    }

    fn temporary_directory() -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "turbospark-image-metal-{}-{timestamp}",
            std::process::id()
        ))
    }
}
