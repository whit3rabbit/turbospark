//! Device execution seam for dense LM backbones.
//!
//! With a [`ComputeBackend`] supplied by the runtime (e.g. Metal on macOS),
//! the backbone offloads linear projections and non-causal attention to the
//! device. When no backend is supplied, execution stays entirely on the host
//! CPU.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::backend::{
    AttentionShape, ComputeBackend, DType, DeviceWeight, WeightData, WeightEncoding,
};
use crate::nn::Linear;
use crate::Result;

#[derive(Default)]
pub(crate) struct Execution {
    backend: Option<Rc<dyn ComputeBackend>>,
    weights: RefCell<HashMap<usize, Rc<dyn DeviceWeight>>>,
}

impl Execution {
    pub(crate) fn new(backend: Option<Rc<dyn ComputeBackend>>) -> Self {
        Self {
            backend,
            weights: RefCell::new(HashMap::new()),
        }
    }

    pub(crate) fn on_device(&self) -> bool {
        self.backend.is_some()
    }

    #[allow(dead_code)]
    pub(crate) fn backend(&self) -> Option<&Rc<dyn ComputeBackend>> {
        self.backend.as_ref()
    }

    fn weight(&self, backend: &dyn ComputeBackend, layer: &Linear) -> Result<Rc<dyn DeviceWeight>> {
        let key = layer.weight.as_ptr() as usize;
        if let Some(weight) = self.weights.borrow().get(&key) {
            return Ok(weight.clone());
        }
        let bytes: Vec<u8> = layer.weight.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weight = backend.load_weight(WeightData {
            name: "lm_linear",
            shape: &[layer.output, layer.input],
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

    pub(crate) fn linear(&self, layer: &Linear, x: &[f32], rows: usize) -> Vec<f32> {
        if let Some(backend) = &self.backend {
            if let Ok(w) = self.weight(backend.as_ref(), layer) {
                if let Ok(out) = w.linear(
                    x,
                    layer.bias.as_deref(),
                    rows,
                    layer.input,
                    layer.output,
                    DType::F32,
                ) {
                    return out;
                }
            }
        }
        layer.forward(x, rows)
    }

    pub(crate) fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        len: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
    ) -> Result<Vec<f32>> {
        let backend = self
            .backend
            .as_ref()
            .expect("device attention needs a backend");
        let shape = AttentionShape {
            batch: 1,
            queries: len,
            keys: len,
            heads,
            kv_heads,
            dim,
            kv_time_major: false,
            causal: false,
            offset: 0,
        };
        backend.attention(q, k, v, shape, DType::F32)
    }
}
