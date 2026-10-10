//! Several Music 3 operations encoded into one command buffer and read back
//! once. Projections, norms, and attention go through the same plans
//! (kernel choice and parameters) as their synchronous forms in the parent
//! module, so each output is bit-identical to the per-operation result; the
//! only difference is that intermediates stay on the device.
use super::{
    attention_plan, bad, bytes, encode_attention, encode_norm, encode_unary, norm_plan, product,
    rope_params, Music3DType, Music3Device, Music3Weight, State, SEQUENCE_SOURCE,
};
use crate::{autorelease_pool, read_f32_buffer_at, GpuError, MetalBuffer, PassEncoder};
use std::rc::Rc;

/// A tensor of a [`Music3Sequence`]: an upload, an operation output, or a
/// view into one of those.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Music3Slot(usize);

/// A resident f32 lookup table, such as the BF16 SiLU activation table.
pub struct Music3Table {
    state: Rc<State>,
    buffer: MetalBuffer,
    len: usize,
}

/// A persistent device f32 buffer that outlives one sequence (the AR KV
/// cache). Shared storage: the host writes and reads it directly between
/// submissions.
pub struct Music3Buffer {
    state: Rc<State>,
    buffer: MetalBuffer,
    len: usize,
}

impl Music3Buffer {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Host write at element `first`. Call only while no submitted sequence
    /// that binds this buffer is running.
    pub fn write(&self, first: usize, data: &[f32]) -> Result<(), GpuError> {
        if first
            .checked_add(data.len())
            .is_none_or(|end| end > self.len)
        {
            return Err(bad("Music 3 buffer write is out of range"));
        }
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        crate::write_buffer_bytes(&self.buffer, first * 4, &bytes);
        Ok(())
    }
    /// Host read of `len` elements from element `first`.
    pub fn read(&self, first: usize, len: usize) -> Result<Vec<f32>, GpuError> {
        if first.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(bad("Music 3 buffer read is out of range"));
        }
        Ok(read_f32_buffer_at(&self.buffer, first, len))
    }
}

struct Tensor {
    buffer: usize,
    /// Offset in f32 elements.
    first: usize,
    len: usize,
}

/// Elements in the BF16 SiLU table: one entry per 16-bit pattern.
const SILU_TABLE_LEN: usize = 1 << 16;

pub struct Music3Sequence {
    state: Rc<State>,
    pass: Option<PassEncoder>,
    /// Every buffer the pass touches, kept alive until it completes.
    buffers: Vec<MetalBuffer>,
    tensors: Vec<Tensor>,
    /// Operation outputs, scanned for non-finite values before readback.
    outputs: Vec<Music3Slot>,
    /// Bound for the unused bias of projections and norms.
    zero: usize,
    /// `[non-finite, inexact]` flags.
    flags: usize,
}

impl Music3Device {
    /// Upload a lookup table once; it stays resident while referenced.
    pub fn table(&self, values: &[f32]) -> Result<Music3Table, GpuError> {
        if values.is_empty() {
            return Err(bad("Music 3 table is empty"));
        }
        autorelease_pool(|| {
            let buffer = self.state.context.borrow().new_buffer_with_data(values);
            Ok(Music3Table {
                state: self.state.clone(),
                buffer,
                len: values.len(),
            })
        })
    }
    /// A persistent buffer of `len` f32 elements. Contents start undefined.
    pub fn buffer(&self, len: usize) -> Result<Music3Buffer, GpuError> {
        let bytes = product(&[len, 4])?;
        if len == 0 {
            return Err(bad("Music 3 buffer is empty"));
        }
        autorelease_pool(|| {
            let buffer = self.state.context.borrow().new_output_buffer(bytes as u64);
            Ok(Music3Buffer {
                state: self.state.clone(),
                buffer,
                len,
            })
        })
    }
    /// Open one command buffer named `label` for a sequence of operations.
    pub fn sequence(&self, label: &'static str) -> Music3Sequence {
        autorelease_pool(|| {
            let c = self.state.context.borrow();
            let zero = c.new_buffer_with_data(&[0.0f32]);
            let flags = c.new_buffer_with_data(&[0u32; 2]);
            let pass = c.begin_pass_labeled(label);
            Music3Sequence {
                state: self.state.clone(),
                pass: Some(pass),
                buffers: vec![zero, flags],
                tensors: Vec::new(),
                outputs: Vec::new(),
                zero: 0,
                flags: 1,
            }
        })
    }
}

impl Music3Sequence {
    fn tensor(&self, slot: Music3Slot) -> Result<&Tensor, GpuError> {
        self.tensors
            .get(slot.0)
            .ok_or_else(|| bad("Music 3 sequence slot does not exist"))
    }
    /// `(buffer, byte offset)` binding of a slot.
    fn binding(&self, slot: Music3Slot) -> Result<(&MetalBuffer, u64), GpuError> {
        let t = self.tensor(slot)?;
        Ok((&self.buffers[t.buffer], (t.first * 4) as u64))
    }
    fn len(&self, slot: Music3Slot) -> Result<usize, GpuError> {
        Ok(self.tensor(slot)?.len)
    }
    fn push(&mut self, buffer: MetalBuffer, len: usize) -> Music3Slot {
        self.buffers.push(buffer);
        self.tensors.push(Tensor {
            buffer: self.buffers.len() - 1,
            first: 0,
            len,
        });
        Music3Slot(self.tensors.len() - 1)
    }
    fn output(&mut self, len: usize) -> Music3Slot {
        let buffer = self
            .state
            .context
            .borrow()
            .new_output_buffer((len * 4) as u64);
        let slot = self.push(buffer, len);
        self.outputs.push(slot);
        slot
    }
    fn pass(&self) -> Result<&PassEncoder, GpuError> {
        self.pass
            .as_ref()
            .ok_or_else(|| bad("Music 3 sequence already finished"))
    }

    pub fn upload(&mut self, data: &[f32]) -> Result<Music3Slot, GpuError> {
        if data.is_empty() {
            return Err(bad("Music 3 sequence upload is empty"));
        }
        autorelease_pool(|| {
            let buffer = self.state.context.borrow().new_buffer_with_data(data);
            Ok(self.push(buffer, data.len()))
        })
    }
    /// A persistent buffer as a slot (no copy). Views of it address rows.
    pub fn bind(&mut self, buffer: &Music3Buffer) -> Result<Music3Slot, GpuError> {
        if !Rc::ptr_eq(&buffer.state, &self.state) {
            return Err(bad("Music 3 buffer belongs to another device"));
        }
        Ok(self.push(buffer.buffer.clone(), buffer.len))
    }
    /// Copy `src` into `dst` bit for bit (both already exist; `dst` is
    /// usually a view of a bound buffer).
    pub fn copy(&mut self, src: Music3Slot, dst: Music3Slot) -> Result<(), GpuError> {
        let len = self.len(src)?;
        if self.len(dst)? != len {
            return Err(bad("Music 3 sequence copy operands differ in length"));
        }
        let count = u32::try_from(len).map_err(|_| bad("Music 3 length overflows"))?;
        autorelease_pool(|| {
            let mut c = self.state.context.borrow_mut();
            let pipeline = c.pipeline(
                SEQUENCE_SOURCE,
                "music3_copy",
                &crate::rms_norm::unused_function_constants(),
                b"",
            )?;
            let (sb, so) = self.binding(src)?;
            let (db, d_off) = self.binding(dst)?;
            let params = bytes(&[count]);
            self.pass()?.encode_threads_3d(
                &pipeline,
                &[(sb, 0, so), (db, 1, d_off)],
                &[(&params, 2)],
                (len as u64, 1, 1),
                (128, 1, 1),
            );
            Ok(())
        })
    }
    /// NeoX rope with the synchronous `Music3Device::rope` parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &mut self,
        x: Music3Slot,
        batch: usize,
        seq: usize,
        heads: usize,
        dim: usize,
        offset: usize,
        theta: f32,
        dtype: Music3DType,
    ) -> Result<Music3Slot, GpuError> {
        let len = self.len(x)?;
        let (params, threads) = rope_params(len, batch, seq, heads, dim, offset, theta, dtype)?;
        autorelease_pool(|| {
            let out = self.output(len);
            let mut c = self.state.context.borrow_mut();
            encode_unary(
                &mut c,
                self.pass()?,
                "music3_rope",
                dtype,
                &params,
                threads,
                self.binding(x)?,
                self.binding(out)?.0,
            )?;
            Ok(out)
        })
    }
    /// `len` elements of `slot` starting at element `first`, without a copy.
    /// The start stays 16-byte aligned so vector loads remain aligned.
    pub fn view(
        &mut self,
        slot: Music3Slot,
        first: usize,
        len: usize,
    ) -> Result<Music3Slot, GpuError> {
        let t = self.tensor(slot)?;
        let start = t.first + first;
        if len == 0 || first.checked_add(len).is_none_or(|end| end > t.len) || start % 4 != 0 {
            return Err(bad("Music 3 sequence view is out of range or unaligned"));
        }
        let buffer = t.buffer;
        self.tensors.push(Tensor {
            buffer,
            first: start,
            len,
        });
        Ok(Music3Slot(self.tensors.len() - 1))
    }

    /// A projection with the synchronous call's plan. `bias` is added in the
    /// kernel, as `Music3Weight::linear_typed` does.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &mut self,
        x: Music3Slot,
        weight: &Music3Weight,
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        dtype: Music3DType,
    ) -> Result<Music3Slot, GpuError> {
        if !Rc::ptr_eq(&weight.state, &self.state) {
            return Err(bad("Music 3 weight belongs to another device"));
        }
        let plan = weight.linear_plan(
            self.len(x)?,
            bias.map(<[f32]>::len),
            rows,
            input_dim,
            output_dim,
            dtype,
            false,
        )?;
        autorelease_pool(|| {
            let bias = bias.map(|b| self.upload(b)).transpose()?;
            let out = self.output(plan.count);
            let mut c = self.state.context.borrow_mut();
            let (xb, xo) = self.binding(x)?;
            let (ob, _) = self.binding(out)?;
            let bb = match bias {
                Some(slot) => self.binding(slot)?.0,
                None => &self.buffers[self.zero],
            };
            weight.encode_linear(&mut c, self.pass()?, &plan, (xb, xo), bb, ob)?;
            Ok(out)
        })
    }

    pub fn rms_norm(
        &mut self,
        x: Music3Slot,
        w: &[f32],
        rows: usize,
        cols: usize,
        eps: f32,
        dtype: Music3DType,
    ) -> Result<Music3Slot, GpuError> {
        let plan = norm_plan(
            self.len(x)?,
            w.len(),
            None,
            rows,
            cols,
            eps,
            dtype,
            false,
            false,
        )?;
        autorelease_pool(|| {
            let weight = self.upload(w)?;
            let out = self.output(plan.count);
            let mut c = self.state.context.borrow_mut();
            let (xb, xo) = self.binding(x)?;
            let (wb, _) = self.binding(weight)?;
            let (ob, _) = self.binding(out)?;
            encode_norm(
                &mut c,
                self.pass()?,
                &plan,
                (xb, xo),
                wb,
                &self.buffers[self.zero],
                ob,
            )
            .map(|()| out)
        })
    }

    /// Layer norm with an optional bias, with the synchronous plan.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &mut self,
        x: Music3Slot,
        w: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        cols: usize,
        eps: f32,
        dtype: Music3DType,
    ) -> Result<Music3Slot, GpuError> {
        let plan = norm_plan(
            self.len(x)?,
            w.len(),
            bias.map(<[f32]>::len),
            rows,
            cols,
            eps,
            dtype,
            true,
            false,
        )?;
        autorelease_pool(|| {
            let weight = self.upload(w)?;
            let bias = bias.map(|b| self.upload(b)).transpose()?;
            let out = self.output(plan.count);
            let mut c = self.state.context.borrow_mut();
            let (xb, xo) = self.binding(x)?;
            let (wb, _) = self.binding(weight)?;
            let bb = match bias {
                Some(slot) => self.binding(slot)?.0,
                None => &self.buffers[self.zero],
            };
            let (ob, _) = self.binding(out)?;
            encode_norm(&mut c, self.pass()?, &plan, (xb, xo), wb, bb, ob).map(|()| out)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        q: Music3Slot,
        k: Music3Slot,
        v: Music3Slot,
        batch: usize,
        queries: usize,
        keys: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        time_major: bool,
        causal: bool,
        offset: usize,
        dtype: Music3DType,
    ) -> Result<Music3Slot, GpuError> {
        let plan = attention_plan(
            [self.len(q)?, self.len(k)?, self.len(v)?],
            batch,
            queries,
            keys,
            heads,
            kv_heads,
            dim,
            time_major,
            causal,
            offset,
            dtype,
        )?;
        autorelease_pool(|| {
            let out = self.output(plan.count);
            let scratch = {
                let mut c = self.state.context.borrow_mut();
                encode_attention(
                    &mut c,
                    self.pass()?,
                    &plan,
                    self.binding(q)?,
                    self.binding(k)?,
                    self.binding(v)?,
                    self.binding(out)?.0,
                )?
            };
            self.buffers.extend(scratch);
            Ok(out)
        })
    }

    /// `Bf16.round(a + b)`, the host residual add.
    pub fn add_round_bf16(&mut self, a: Music3Slot, b: Music3Slot) -> Result<Music3Slot, GpuError> {
        let len = self.len(a)?;
        if self.len(b)? != len {
            return Err(bad("Music 3 sequence add operands differ in length"));
        }
        self.elementwise("music3_add_round_bf16", &[a, b], len, &[])
    }

    /// `Bf16.round(table[g >> 16] * u)`, the host SwiGLU with `table` the
    /// host BF16 SiLU of every 16-bit pattern.
    pub fn swiglu_bf16(
        &mut self,
        g: Music3Slot,
        u: Music3Slot,
        table: &Music3Table,
    ) -> Result<Music3Slot, GpuError> {
        if !Rc::ptr_eq(&table.state, &self.state) || table.len != SILU_TABLE_LEN {
            return Err(bad("Music 3 SiLU table does not belong to this device"));
        }
        let len = self.len(g)?;
        if self.len(u)? != len {
            return Err(bad("Music 3 sequence SwiGLU operands differ in length"));
        }
        let silu = self.push(table.buffer.clone(), table.len);
        self.elementwise("music3_swiglu_bf16", &[g, u, silu], len, &[])
    }

    /// The host BF16 partial NeoX rotary of the DiT over `[len, heads *
    /// head_dim]`, rotating the leading `rotary` features of every head with
    /// `[len, rotary / 2]` tables.
    #[allow(clippy::too_many_arguments)]
    pub fn partial_rope_bf16(
        &mut self,
        x: Music3Slot,
        cos: Music3Slot,
        sin: Music3Slot,
        len: usize,
        heads: usize,
        head_dim: usize,
        rotary: usize,
    ) -> Result<Music3Slot, GpuError> {
        let inner = product(&[heads, head_dim])?;
        let half = rotary / 2;
        if self.len(x)? != product(&[len, inner])?
            || rotary % 2 != 0
            || rotary > head_dim
            || self.len(cos)? != product(&[len, half])?
            || self.len(sin)? != len * half
        {
            return Err(bad("Music 3 partial rotary geometry mismatch"));
        }
        let extra = [inner as u32, head_dim as u32, half as u32];
        self.elementwise(
            "music3_partial_rope_bf16",
            &[x, cos, sin],
            len * inner,
            &extra,
        )
    }

    /// `round(state * silu(gate))` over rows of `[state | gate]`, each `half`
    /// wide, with the host BF16 SiLU table.
    pub fn gated_silu_bf16(
        &mut self,
        x: Music3Slot,
        rows: usize,
        half: usize,
        table: &Music3Table,
    ) -> Result<Music3Slot, GpuError> {
        if !Rc::ptr_eq(&table.state, &self.state) || table.len != SILU_TABLE_LEN {
            return Err(bad("Music 3 SiLU table does not belong to this device"));
        }
        if self.len(x)? != product(&[rows, half, 2])? {
            return Err(bad("Music 3 gated activation geometry mismatch"));
        }
        let silu = self.push(table.buffer.clone(), table.len);
        self.elementwise(
            "music3_gated_silu_bf16",
            &[x, silu],
            rows * half,
            &[half as u32],
        )
    }

    /// One thread per element: inputs at 0.., then the output, then the
    /// inexact flag, then the element count followed by `extra`.
    fn elementwise(
        &mut self,
        kernel: &'static str,
        inputs: &[Music3Slot],
        len: usize,
        extra: &[u32],
    ) -> Result<Music3Slot, GpuError> {
        let count = u32::try_from(product(&[len])?).map_err(|_| bad("Music 3 length overflows"))?;
        autorelease_pool(|| {
            let out = self.output(len);
            let mut c = self.state.context.borrow_mut();
            let pipeline = c.pipeline(
                SEQUENCE_SOURCE,
                kernel,
                &crate::rms_norm::unused_function_constants(),
                b"",
            )?;
            let mut bindings = Vec::with_capacity(inputs.len() + 2);
            for (index, &slot) in inputs.iter().enumerate() {
                let (buffer, offset) = self.binding(slot)?;
                bindings.push((buffer, index as u64, offset));
            }
            bindings.push((self.binding(out)?.0, inputs.len() as u64, 0));
            bindings.push((&self.buffers[self.flags], inputs.len() as u64 + 1, 4));
            let params = bytes(&[&[count][..], extra].concat());
            self.pass()?.encode_threads_3d(
                &pipeline,
                &bindings,
                &[(&params, inputs.len() as u64 + 2)],
                (len as u64, 1, 1),
                (128, 1, 1),
            );
            Ok(out)
        })
    }

    /// Scan every operation output, commit, wait, and read `read` back.
    /// `Ok(None)` means an elementwise step met a subnormal case the device
    /// may round differently from the host: recompute on the per-operation
    /// path. Non-finite values are an error, as in the synchronous path.
    pub fn finish(mut self, read: &[Music3Slot]) -> Result<Option<Vec<Vec<f32>>>, GpuError> {
        for &slot in read {
            self.tensor(slot)?;
        }
        autorelease_pool(|| {
            {
                let mut c = self.state.context.borrow_mut();
                let pipeline = c.pipeline(
                    SEQUENCE_SOURCE,
                    "music3_nonfinite",
                    &crate::rms_norm::unused_function_constants(),
                    b"",
                )?;
                for &slot in &self.outputs {
                    let (buffer, offset) = self.binding(slot)?;
                    let len = self.len(slot)?;
                    let params = bytes(&[len as u32]);
                    self.pass()?.encode_threads_3d(
                        &pipeline,
                        &[(buffer, 0, offset), (&self.buffers[self.flags], 1, 0)],
                        &[(&params, 2)],
                        (len as u64, 1, 1),
                        (128, 1, 1),
                    );
                }
            }
            let pass = self
                .pass
                .take()
                .ok_or_else(|| bad("Music 3 sequence already finished"))?;
            pass.commit_and_wait_checked()?;
            let flags = read_u32_pair(&self.buffers[self.flags]);
            if flags[0] != 0 {
                return Err(bad("Music 3 device produced non-finite values"));
            }
            if flags[1] != 0 {
                return Ok(None);
            }
            read.iter()
                .map(|&slot| {
                    let t = self.tensor(slot)?;
                    let values = read_f32_buffer_at(&self.buffers[t.buffer], t.first, t.len);
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err(bad("Music 3 device produced non-finite values"));
                    }
                    Ok(values)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
        })
    }
}

fn read_u32_pair(buffer: &MetalBuffer) -> [u32; 2] {
    let raw = crate::read_buffer_bytes(buffer, 0, 8);
    [
        u32::from_le_bytes(raw[..4].try_into().unwrap()),
        u32::from_le_bytes(raw[4..].try_into().unwrap()),
    ]
}
