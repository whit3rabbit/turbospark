//! Shared backend dispatch, with immutable weights cached for this model only.
use crate::backend::{AttentionShape, ConvShape};
use crate::backend::{ComputeBackend, DType, DeviceWeight, WeightData, WeightEncoding};
use crate::{ops, Result, SpeechError};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Default)]
pub(super) struct Execution {
    backend: Option<Rc<dyn ComputeBackend>>,
    weights: RefCell<HashMap<(usize, Vec<usize>), Rc<dyn DeviceWeight>>>,
}
impl Execution {
    pub(super) fn new(backend: Option<Rc<dyn ComputeBackend>>) -> Self {
        Self {
            backend,
            weights: RefCell::new(HashMap::new()),
        }
    }
    fn weight(&self, values: &[f32], shape: &[usize]) -> Result<Rc<dyn DeviceWeight>> {
        // Kokoro's vectors remain immutable and live as long as this cache.
        let key = (values.as_ptr() as usize, shape.to_vec());
        if let Some(weight) = self.weights.borrow().get(&key) {
            return Ok(weight.clone());
        }
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weight = self
            .backend
            .as_ref()
            .expect("device path")
            .load_weight(WeightData {
                name: "kokoro",
                shape,
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
    pub(super) fn linear(
        &self,
        x: &[f32],
        w: &[f32],
        b: Option<&[f32]>,
        rows: usize,
        input: usize,
        output: usize,
    ) -> Result<Vec<f32>> {
        if self.backend.is_some() {
            self.weight(w, &[output, input])?
                .linear(x, b, rows, input, output, DType::F32)
        } else {
            Ok(ops::linear(x, w, b, rows, input, output))
        }
    }
    pub(super) fn normalize(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
        layout: crate::backend::NormalizationLayout,
    ) -> Result<Option<Vec<f32>>> {
        self.backend
            .as_ref()
            .map(|b| b.normalize(x, rows, cols, 1e-5, layout))
            .transpose()
    }
    pub(super) fn sine(&self, x: &[f32]) -> Result<Vec<f32>> {
        match &self.backend {
            Some(b) => b.sine(x),
            None => Ok(x.iter().map(|v| v.sin()).collect()),
        }
    }
    pub(super) fn tanh(&self, x: &[f32]) -> Result<Vec<f32>> {
        match &self.backend {
            Some(b) => b.tanh(x),
            None => Ok(x.iter().map(|v| v.tanh()).collect()),
        }
    }
    pub(super) fn using_device(&self) -> bool {
        self.backend.is_some()
    }
    pub(super) fn layer_norm(
        &self,
        x: &mut Vec<f32>,
        w: &[f32],
        b: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        if let Some(backend) = &self.backend {
            *x = backend.layer_norm(x, w, Some(b), rows, cols, eps, DType::F32)?;
        } else {
            ops::layernorm(x, rows, cols, w, Some(b), eps);
        }
        Ok(())
    }
    pub(super) fn gelu(&self, x: &mut Vec<f32>) -> Result<()> {
        if let Some(backend) = &self.backend {
            *x = backend.gelu_erf(x)?;
        } else {
            ops::gelu_erf(x);
        }
        Ok(())
    }
    pub(super) fn weight_norm(
        &self,
        x: Vec<f32>,
        g: &[f32],
        rows: usize,
        input: usize,
        kernel: usize,
    ) -> Result<Vec<f32>> {
        if let Some(backend) = &self.backend {
            backend.weight_norm(&x, g, rows, input, kernel)
        } else {
            Ok(super::weight_norm_oki(x, g, rows, input, kernel))
        }
    }
    pub(super) fn embedding(&self, w: &[f32], width: usize, ids: &[u32]) -> Result<Vec<f32>> {
        let ids: Vec<i32> = ids
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| SpeechError::Input {
                    why: "phoneme ID exceeds signed 32-bit device contract".into(),
                })
            })
            .collect::<Result<_>>()?;
        if width == 0
            || w.len() % width != 0
            || ids.is_empty()
            || ids
                .iter()
                .any(|&id| id < 0 || id as usize >= w.len() / width)
        {
            return Err(SpeechError::Input {
                why: "phoneme ID does not fit embedding table".into(),
            });
        }
        if self.backend.is_some() {
            self.weight(w, &[w.len() / width, width])?
                .embedding(&ids, width)
        } else {
            Ok(ops::embedding(w, width, &ids))
        }
    }
    pub(super) fn convolution(
        &self,
        x: &[f32],
        w: &[f32],
        b: Option<&[f32]>,
        shape: ConvShape,
        groups: usize,
    ) -> Result<Vec<f32>> {
        if self.backend.is_some() {
            let dims = if shape.transpose {
                vec![
                    shape.input_channels,
                    shape.output_channels / groups,
                    shape.kernel,
                ]
            } else {
                vec![
                    shape.output_channels,
                    shape.input_channels / groups,
                    shape.kernel,
                ]
            };
            self.weight(w, &dims)?
                .convolution_grouped(x, b, shape, groups, DType::F32)
        } else if shape.transpose {
            Ok(ops::conv_transpose1d(
                x,
                w,
                b,
                shape.input_channels,
                shape.output_channels,
                shape.kernel,
                shape.stride,
                shape.padding,
                0,
                groups,
            ))
        } else {
            Ok(ops::conv1d(
                x,
                w,
                b,
                shape.input_channels,
                shape.output_channels,
                shape.kernel,
                shape.stride,
                shape.padding,
                shape.dilation,
                groups,
            ))
        }
    }
    pub(super) fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        dim: usize,
    ) -> Result<Vec<f32>> {
        match &self.backend {
            Some(backend) => backend.attention(
                q,
                k,
                v,
                AttentionShape {
                    batch: 1,
                    queries: seq,
                    keys: seq,
                    heads: 1,
                    kv_heads: 1,
                    dim,
                    kv_time_major: false,
                    causal: false,
                    offset: 0,
                },
                DType::F32,
            ),
            None => Ok(ops::sdpa(
                q,
                k,
                v,
                None,
                seq,
                seq,
                dim,
                dim,
                (dim as f32).sqrt().recip(),
            )),
        }
    }
    pub(super) fn recurrence(
        &self,
        projection: &[f32],
        weights: &[f32],
        hidden: usize,
        backward: bool,
    ) -> Result<Option<Vec<f32>>> {
        if self.backend.is_some() {
            self.weight(weights, &[4 * hidden, hidden])?
                .lstm_recurrence(projection, hidden, backward)
                .map(Some)
        } else {
            Ok(None)
        }
    }
    pub(super) fn normal(&self, uniforms: &[f32]) -> Result<Vec<f32>> {
        match &self.backend {
            Some(b) => b.normal_from_uniform(uniforms, DType::F32),
            None => Ok(crate::music::minimax_music3::rng::normal_from_uniform(
                uniforms,
                DType::F32,
            )),
        }
    }
    pub(super) fn cumulative_sum(
        &self,
        x: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>> {
        if let Some(backend) = &self.backend {
            return backend.cumulative_sum(x, rows, columns);
        }
        let mut out = x.to_vec();
        for column in 0..columns {
            let mut sum = 0.0;
            for row in 0..rows {
                sum += x[row * columns + column];
                out[row * columns + column] = sum;
            }
        }
        Ok(out)
    }
    pub(super) fn stft(&self, x: &[f32], window: &[f32], hop: usize) -> Result<Option<Vec<f32>>> {
        match &self.backend {
            Some(backend) => backend.stft_magnitude_phase(x, window, hop).map(Some),
            None => Ok(None),
        }
    }
    pub(super) fn trace(&self, stage: &str, values: &[f32], shape: &[usize]) {
        if let Some(b) = &self.backend {
            b.trace(stage, values, DType::F32, shape);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{AttentionShape, ConvShape};
    use std::cell::Cell;
    struct Device(Cell<usize>);
    struct Matrix(Vec<f32>);
    impl DeviceWeight for Matrix {
        fn linear(
            &self,
            x: &[f32],
            b: Option<&[f32]>,
            r: usize,
            i: usize,
            o: usize,
            _: DType,
        ) -> Result<Vec<f32>> {
            Ok(ops::linear(x, &self.0, b, r, i, o))
        }
        fn embedding(&self, _: &[i32], _: usize) -> Result<Vec<f32>> {
            unreachable!()
        }
        fn convolution(
            &self,
            _: &[f32],
            _: Option<&[f32]>,
            _: ConvShape,
            _: DType,
        ) -> Result<Vec<f32>> {
            unreachable!()
        }
    }
    impl ComputeBackend for Device {
        fn load_weight(&self, d: WeightData<'_>) -> Result<Rc<dyn DeviceWeight>> {
            self.0.set(self.0.get() + 1);
            Ok(Rc::new(Matrix(
                d.bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect(),
            )))
        }
        fn attention(
            &self,
            _: &[f32],
            _: &[f32],
            _: &[f32],
            _: AttentionShape,
            _: DType,
        ) -> Result<Vec<f32>> {
            unreachable!()
        }
    }
    #[test]
    fn kokoro_device_dispatch_caches_resident_matrix() {
        let device = Rc::new(Device(Cell::new(0)));
        let execution = Execution::new(Some(device.clone()));
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        for _ in 0..2 {
            assert_eq!(
                execution
                    .linear(&[2.0, -1.0], &weights, Some(&[0.5, -0.5]), 1, 2, 2)
                    .unwrap(),
                vec![0.5, 1.5]
            );
        }
        assert_eq!(
            device.0.get(),
            1,
            "Kokoro heavy projection must use the device and reuse its resident weight"
        );
    }
}
