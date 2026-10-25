//! Where the Fish S1 DAC's heavy operations run.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::backend::{ComputeBackend, ConvShape, DType, DeviceWeight, WeightData, WeightEncoding};
use crate::codec::conv::{Conv1d, ConvTranspose1d};
use crate::codec::wnconv::snake1d;
use crate::ops;
use crate::Result;

/// The device a [`FishS1Dac`](super::FishS1Dac) decode or encode runs its
/// convolutions, transposed convolutions, ConvNeXt pointwise linears and
/// Snake on, or the host. Code lookups, the transformers, norms, GELU,
/// causal padding and trimming stay on the host.
///
/// An exec lives apart from the codec so the codec stays `Send` and `Sync`
/// for the families that share it, while a device backend (tied to one
/// thread) and its uploaded weights live with the thread that decodes.
/// Uploaded weights are cached by host address, so one exec serves one codec
/// for that codec's lifetime.
#[derive(Default)]
pub struct DacExec {
    backend: Option<Rc<dyn ComputeBackend>>,
    weights: RefCell<HashMap<usize, Rc<dyn DeviceWeight>>>,
}

impl DacExec {
    /// Every operation on the host.
    pub fn host() -> Self {
        Self::default()
    }

    /// The heavy operations on `backend`, F32 weights and activations.
    pub fn device(backend: Rc<dyn ComputeBackend>) -> Self {
        DacExec {
            backend: Some(backend),
            weights: RefCell::default(),
        }
    }

    pub fn is_device(&self) -> bool {
        self.backend.is_some()
    }

    fn weight(
        &self,
        backend: &dyn ComputeBackend,
        values: &[f32],
        dims: &[usize],
    ) -> Result<Rc<dyn DeviceWeight>> {
        let key = values.as_ptr() as usize;
        if let Some(weight) = self.weights.borrow().get(&key) {
            return Ok(weight.clone());
        }
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weight = backend.load_weight(WeightData {
            name: "fish_s1_dac",
            shape: dims,
            bytes: &bytes,
            encoding: WeightEncoding::F32,
            dtype: DType::F32,
            scales_dtype: None,
            offsets_dtype: None,
            scales: &[],
            offsets: &[],
            block_scales: &[],
            planned_rows: None,
        })?;
        self.weights.borrow_mut().insert(key, weight.clone());
        Ok(weight)
    }

    /// [`Conv1d::forward`], on the device when there is one.
    pub(super) fn conv(&self, c: &Conv1d, x: &[f32]) -> Result<Vec<f32>> {
        let Some(backend) = &self.backend else {
            return Ok(c.forward(x));
        };
        let shape = ConvShape {
            input_channels: c.in_ch,
            output_channels: c.out_ch,
            kernel: c.kernel,
            stride: c.stride,
            padding: c.padding,
            dilation: c.dilation,
            transpose: false,
        };
        self.weight(
            backend.as_ref(),
            &c.weight,
            &[c.out_ch, c.in_ch / c.groups, c.kernel],
        )?
        .convolution_grouped(x, c.bias.as_deref(), shape, c.groups, DType::F32)
    }

    /// [`ConvTranspose1d::forward`], on the device when there is one. The
    /// device contract has no output padding (this codec uses none).
    pub(super) fn conv_transpose(&self, c: &ConvTranspose1d, x: &[f32]) -> Result<Vec<f32>> {
        match &self.backend {
            Some(backend) if c.output_padding == 0 => {
                let shape = ConvShape {
                    input_channels: c.in_ch,
                    output_channels: c.out_ch,
                    kernel: c.kernel,
                    stride: c.stride,
                    padding: c.padding,
                    dilation: 1,
                    transpose: true,
                };
                self.weight(
                    backend.as_ref(),
                    &c.weight,
                    &[c.in_ch, c.out_ch / c.groups, c.kernel],
                )?
                .convolution_grouped(
                    x,
                    c.bias.as_deref(),
                    shape,
                    c.groups,
                    DType::F32,
                )
            }
            _ => Ok(c.forward(x)),
        }
    }

    /// `x [rows, input]` times `w [output, input]` transposed, plus `bias`.
    pub(super) fn linear(
        &self,
        x: &[f32],
        w: &[f32],
        bias: &[f32],
        rows: usize,
        input: usize,
        output: usize,
    ) -> Result<Vec<f32>> {
        let Some(backend) = &self.backend else {
            return Ok(ops::linear(x, w, Some(bias), rows, input, output));
        };
        self.weight(backend.as_ref(), w, &[output, input])?.linear(
            x,
            Some(bias),
            rows,
            input,
            output,
            DType::F32,
        )
    }

    /// [`snake1d`] in place. A device divides by `alpha + 1e-9` where the
    /// host multiplies by its reciprocal, so the two agree to rounding.
    pub(super) fn snake(
        &self,
        x: &mut Vec<f32>,
        alpha: &[f32],
        channels: usize,
        frames: usize,
    ) -> Result<()> {
        match &self.backend {
            Some(backend) => *x = backend.snake(x, alpha, channels, frames, DType::F32)?,
            None => snake1d(x, alpha, channels, frames),
        }
        Ok(())
    }
}
