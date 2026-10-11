//! Optional device dispatch for the DiT's heavy operations: the six
//! per-block projections and attention. Everything else (norms, modulation,
//! RoPE, activations, the small projections) stays on the host, where it is
//! cheap and keeps its reference rounding. The crate stays portable: a
//! runtime supplies the [`ComputeBackend`]; without one every call is the
//! host path the parity tests pin.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::backend::{
    AttentionShape, ComputeBackend, DType, DeviceWeight, WeightData, WeightEncoding,
};
use crate::nn::Linear;
use crate::Result;

#[derive(Default)]
pub(super) struct Execution {
    backend: Option<Rc<dyn ComputeBackend>>,
    /// Uploaded weights keyed by the first host buffer and how many layers
    /// were stacked into them; the host buffers live as long as the model.
    weights: RefCell<HashMap<(usize, usize), Rc<dyn DeviceWeight>>>,
}

impl Execution {
    pub(super) fn new(backend: Option<Rc<dyn ComputeBackend>>) -> Self {
        Self {
            backend,
            weights: RefCell::new(HashMap::new()),
        }
    }

    pub(super) fn on_device(&self) -> bool {
        self.backend.is_some()
    }

    /// `layers` (same input width) uploaded as one `[sum of outputs, input]`
    /// weight. Built only on first use, so no stacked host copy is kept.
    fn weight(
        &self,
        backend: &dyn ComputeBackend,
        layers: &[&Linear],
    ) -> Result<Rc<dyn DeviceWeight>> {
        let key = (layers[0].weight.as_ptr() as usize, layers.len());
        if let Some(weight) = self.weights.borrow().get(&key) {
            return Ok(weight.clone());
        }
        let outputs: usize = layers.iter().map(|l| l.output).sum();
        let bytes: Vec<u8> = layers
            .iter()
            .flat_map(|l| l.weight.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        let weight = backend.load_weight(WeightData {
            name: "megatts3_dit",
            shape: &[outputs, layers[0].input],
            bytes: &bytes,
            encoding: WeightEncoding::F32,
            dtype: DType::F32,
            scales_dtype: None,
            offsets_dtype: None,
            scales: &[],
            offsets: &[],
            block_scales: &[],
        })?;
        self.weights.borrow_mut().insert(key, weight.clone());
        Ok(weight)
    }

    /// `layer` over `rows` row-major inputs, on the device when there is one.
    pub(super) fn linear(&self, layer: &Linear, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        match &self.backend {
            Some(backend) => self.weight(backend.as_ref(), &[layer])?.linear(
                x,
                layer.bias.as_deref(),
                rows,
                layer.input,
                layer.output,
                DType::F32,
            ),
            None => Ok(layer.forward(x, rows)),
        }
    }

    /// Several projections of the same input in one device call (the
    /// attention's q, k and v), returned per layer. Each output column is the
    /// same dot product a separate call computes; one dispatch replaces three.
    pub(super) fn linear_stacked(
        &self,
        layers: &[&Linear],
        x: &[f32],
        rows: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let backend = self
            .backend
            .as_ref()
            .expect("stacked projections are a device path");
        let input = layers[0].input;
        let outputs: usize = layers.iter().map(|l| l.output).sum();
        let bias: Option<Vec<f32>> = if layers.iter().all(|l| l.bias.is_some()) {
            Some(
                layers
                    .iter()
                    .flat_map(|l| l.bias.clone().unwrap_or_default())
                    .collect(),
            )
        } else if layers.iter().all(|l| l.bias.is_none()) {
            None
        } else {
            return Err(crate::SpeechError::Input {
                why: "stacked projections must all have a bias or none".into(),
            });
        };
        let out = self.weight(backend.as_ref(), layers)?.linear(
            x,
            bias.as_deref(),
            rows,
            input,
            outputs,
            DType::F32,
        )?;
        let mut parts: Vec<Vec<f32>> = layers
            .iter()
            .map(|l| Vec::with_capacity(rows * l.output))
            .collect();
        for row in out.chunks_exact(outputs) {
            let mut start = 0;
            for (part, layer) in parts.iter_mut().zip(layers) {
                part.extend_from_slice(&row[start..start + layer.output]);
                start += layer.output;
            }
        }
        Ok(parts)
    }

    /// Non-causal attention over position-major `[batch, frames, heads, dim]`
    /// tensors (the layout the projections produce). Device only.
    #[allow(
        clippy::too_many_arguments,
        reason = "attention kernels carry the tensor shape explicitly"
    )]
    pub(super) fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        frames: usize,
        heads: usize,
        dim: usize,
    ) -> Result<Vec<f32>> {
        let backend = self
            .backend
            .as_ref()
            .expect("device attention needs a backend");
        let shape = AttentionShape {
            batch,
            queries: frames,
            keys: frames,
            heads,
            kv_heads: heads,
            dim,
            kv_time_major: false,
            causal: false,
            offset: 0,
        };
        backend.attention(q, k, v, shape, DType::F32)
    }
}

/// Interleaved RoPE on position-major rows `[batch * frames, heads * dim]`.
/// Each pair rotates with exactly the arithmetic of [`ops::rope_interleaved`],
/// which works on head-major planes; only the addressing differs.
pub(super) fn rope_rows(
    x: &mut [f32],
    frames: usize,
    heads: usize,
    dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = dim / 2;
    for (row, positions) in x.chunks_exact_mut(heads * dim).enumerate() {
        let t = row % frames;
        let c = &cos[t * half..(t + 1) * half];
        let s = &sin[t * half..(t + 1) * half];
        for head in positions.chunks_exact_mut(dim) {
            for (d, pair) in head.chunks_exact_mut(2).enumerate() {
                let a = pair[0];
                let b = pair[1];
                pair[0] = a * c[d] - b * s[d];
                pair[1] = b * c[d] + a * s[d];
            }
        }
    }
}

/// A host [`ComputeBackend`] that runs the crate's own ops behind the device
/// interface, so tests can prove the device-path plumbing (layouts, batching,
/// RoPE placement) bit-exact against the host path.
#[cfg(test)]
pub(super) mod host_backend {
    use super::*;
    use crate::{ops, SpeechError};

    pub(in super::super) struct HostBackend {
        /// Attention calls, which only the device path makes.
        pub(in super::super) attentions: std::cell::Cell<usize>,
    }

    struct HostWeight {
        weight: Vec<f32>,
    }

    impl DeviceWeight for HostWeight {
        fn linear(
            &self,
            input: &[f32],
            bias: Option<&[f32]>,
            rows: usize,
            input_dim: usize,
            output_dim: usize,
            _dtype: DType,
        ) -> Result<Vec<f32>> {
            Ok(ops::linear(
                input,
                &self.weight,
                bias,
                rows,
                input_dim,
                output_dim,
            ))
        }
        fn embedding(&self, _ids: &[i32], _width: usize) -> Result<Vec<f32>> {
            Err(SpeechError::Unsupported {
                why: "unused".into(),
            })
        }
        fn convolution(
            &self,
            _input: &[f32],
            _bias: Option<&[f32]>,
            _shape: crate::backend::ConvShape,
            _dtype: DType,
        ) -> Result<Vec<f32>> {
            Err(SpeechError::Unsupported {
                why: "unused".into(),
            })
        }
    }

    impl ComputeBackend for HostBackend {
        fn load_weight(&self, data: WeightData<'_>) -> Result<Rc<dyn DeviceWeight>> {
            let weight = data
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(Rc::new(HostWeight { weight }))
        }
        fn attention(
            &self,
            q: &[f32],
            k: &[f32],
            v: &[f32],
            s: AttentionShape,
            _dtype: DType,
        ) -> Result<Vec<f32>> {
            self.attentions.set(self.attentions.get() + 1);
            // Per batch row: to head-major planes, the host kernel, and back.
            let row = s.queries * s.heads * s.dim;
            let mut out = Vec::with_capacity(q.len());
            for b in 0..s.batch {
                let span = b * row..(b + 1) * row;
                let plane =
                    |x: &[f32]| ops::split_heads(&x[span.clone()], s.queries, s.heads, s.dim);
                let a = ops::mha(
                    &plane(q),
                    &plane(k),
                    &plane(v),
                    None,
                    s.heads,
                    s.queries,
                    s.keys,
                    s.dim,
                    1.0 / (s.dim as f32).sqrt(),
                );
                out.extend(ops::merge_heads(&a, s.queries, s.heads, s.dim));
            }
            Ok(out)
        }
    }

    impl HostBackend {
        pub(in super::super) fn new() -> Rc<Self> {
            Rc::new(Self {
                attentions: std::cell::Cell::new(0),
            })
        }
    }
}
