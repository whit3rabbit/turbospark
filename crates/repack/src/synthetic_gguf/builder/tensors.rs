//! Specialized tensor generator methods for GgufBuilder.

use super::{GgufBuilder, PendingTensor};

impl GgufBuilder {
    /// Push a tensor with caller-supplied packed bytes. `dims` are in GGUF's
    /// own fastest-varying-first order, so a logical `[rows, cols]` matrix
    /// is pushed as `[cols, rows]`.
    pub fn tensor(mut self, name: &str, ggml_type: u32, dims: &[u64], data: Vec<u8>) -> Self {
        self.tensors.push(PendingTensor {
            name: name.to_string(),
            ggml_type,
            dims: dims.to_vec(),
            data,
        });
        self
    }

    /// Push a tensor of `blocks` Q8_0 blocks (32 elements, 34 bytes each),
    /// filled with a deterministic pattern derived from `seed` so two
    /// fixtures with different seeds cannot accidentally compare equal.
    pub fn q8_0_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        assert!(
            elements % 32 == 0,
            "{name}: {elements} elements is not a whole number of 32-element Q8_0 blocks"
        );
        let blocks = elements / 32;
        let mut data = Vec::with_capacity((blocks * 34) as usize);
        for b in 0..blocks {
            // f16 scale, then 32 int8 weights. Two properties matter beyond
            // being deterministic, and both are about what a caller does with
            // the fixture rather than about the format:
            //
            // - the scale is a valid small positive f16 (0.0625) rather than
            //   a random bit pattern, so a dequant reference can consume
            //   these bytes unchanged;
            // - the quants span [-8, 7] rather than the whole byte range, so
            //   the resulting weights are near +/-0.5 and a forward pass
            //   through a fixture install stays inside FP16. At full range
            //   the weights reach +/-32, every sublayer multiplies the
            //   residual by ~250, and a two-layer model overflows to
            //   infinity before it reaches the head - which is a property of
            //   the fixture, not a kernel bug, and cost a debugging round
            //   once already.
            data.extend_from_slice(&0x2C00u16.to_le_bytes());
            for k in 0..32u64 {
                let n = seed
                    .wrapping_add((b as u8).wrapping_mul(31))
                    .wrapping_add(k as u8);
                data.push((((n % 16) as i8) - 8) as u8);
            }
        }
        self.tensor(name, 8, dims, data)
    }

    /// Deterministic weights in `[-0.5, 0.5)` for the K-quant helpers below.
    ///
    /// The range is the point, and it is the same trap the Q8_0 helper's
    /// comment records: at full quantizer range a fixture install's residual
    /// grows by orders of magnitude per sublayer and overflows FP16 before
    /// the head, which reads as a kernel bug and is not one.
    fn small_weights(elements: u64, seed: u8) -> Vec<f32> {
        let mut s = (seed as u32).wrapping_mul(2_654_435_761).wrapping_add(11);
        (0..elements)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Push a Q4_K tensor (256-element superblocks, 144 bytes each). Qwen
    /// 3.6's Q4_K_M puts its routed experts and its embedding table here.
    pub fn q4_k_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        assert!(
            dims[0] % 256 == 0,
            "{name}: rows of {} elements do not tile 256-element Q4_K superblocks",
            dims[0]
        );
        let data = compute::quantize_q4_k(&Self::small_weights(elements, seed));
        self.tensor(name, 12, dims, data)
    }

    /// Valid blocks for the types ggml added after the K and IQ families:
    /// Q1_0 (41), TQ2_0 (35), TQ1_0 (34) and NVFP4 (40). Layouts are exact
    /// (the decoders in `compute` are held to ggml's own `to_float`),
    /// encoder quality is not the point: Q1_0 gets a mean-magnitude scale and
    /// sign bits from `weights`, and the others get code bytes derived from
    /// `weights` (every bit pattern is a legal code) under a small scale, so
    /// decoded values stay inside what the Q8_0 oracle twin can hold.
    #[allow(clippy::needless_range_loop)]
    fn ext_bytes(name: &str, ggml_type: u32, dims: &[u64], weights: &[f32]) -> Vec<u8> {
        use foundation::LogitValue as F16;
        let (elems, bytes) = match ggml_type {
            41 => (128usize, 18usize),
            35 => (256, 66),
            34 => (256, 54),
            40 => (64, 36),
            39 => (32, 17),
            42 => (64, 18),
            64 => (256, 46),
            65 => (256, 42),
            66 => (256, 38),
            other => panic!("{name}: ggml type {other} is not an ext block type"),
        };
        assert!(
            dims[0] as usize % elems == 0,
            "{name}: rows of {} elements do not tile {elems}-element blocks of ggml type {ggml_type}",
            dims[0]
        );
        let code_byte = |w: f32, salt: usize| -> u8 {
            // A deterministic byte from the weight's bits, mixed with its slot.
            let bits = w.to_bits();
            (bits ^ (bits >> 11) ^ (salt as u32).wrapping_mul(0x9E37_79B1)) as u8
        };
        let mut out = Vec::with_capacity(weights.len() / elems * bytes);
        for block in weights.chunks_exact(elems) {
            let mean = block.iter().map(|w| w.abs()).sum::<f32>() / elems as f32;
            let d = F16::from_f32(mean.max(0.01)).to_bits().to_le_bytes();
            match ggml_type {
                41 => {
                    out.extend_from_slice(&d);
                    for byte in block.chunks_exact(8) {
                        out.push(
                            byte.iter()
                                .enumerate()
                                .fold(0u8, |acc, (i, w)| acc | (u8::from(*w >= 0.0) << i)),
                        );
                    }
                }
                42 => {
                    out.extend_from_slice(&d);
                    for i in 0..16 {
                        out.push(code_byte(block[i], i));
                    }
                }
                64..=66 => {
                    // Unsloth fork IQ1 types: every byte is a legal index or
                    // scale code, and a small fixed `d` keeps the largest weight
                    // (15 * 1.125 * d) well inside what a whole model holds.
                    out.extend_from_slice(&F16::from_f32(0.02).to_bits().to_le_bytes());
                    for i in 0..bytes - 2 {
                        out.push(code_byte(block[i % elems], i));
                    }
                }
                39 => {
                    // Exponent 118..=121 is scale 2^-10 .. 2^-7: codebook values reach
                    // 12, and a larger scale overflows FP16 through a whole model.
                    out.push(118 + code_byte(block[0], 0) % 4);
                    for i in 0..16 {
                        out.push(code_byte(block[i], i + 3));
                    }
                }
                35 | 34 => {
                    for (i, w) in block.iter().take(bytes - 2).enumerate() {
                        out.push(code_byte(*w, i));
                    }
                    out.extend_from_slice(&d);
                }
                _ => {
                    // Scales 0x20..0x2F decode to 0.125 .. 0.47; bit 7 clear.
                    for i in 0..4 {
                        out.push(0x20 + code_byte(block[i], i) % 0x10);
                    }
                    for (i, w) in block.iter().take(32).enumerate() {
                        out.push(code_byte(*w, i + 7));
                    }
                }
            }
        }
        out
    }

    /// Push a legacy 32-element-block tensor: ggml type 2 (Q4_0), 3 (Q4_1),
    /// 6 (Q5_0) or 7 (Q5_1), encoded by the production quantizer
    /// (`gguf_quantize`, whose output is held to ggml byte for byte). Rows
    /// tile 32 elements, so the default fixture dimensions already fit.
    pub fn legacy_tensor(self, name: &str, ggml_type: u32, dims: &[u64], seed: u8) -> Self {
        use crate::gguf_quantize::Target;
        if matches!(ggml_type, 34 | 35 | 39 | 40 | 41 | 42 | 64 | 65 | 66) {
            let elements: u64 = dims.iter().product();
            let data = Self::ext_bytes(name, ggml_type, dims, &Self::small_weights(elements, seed));
            return self.tensor(name, ggml_type, dims, data);
        }
        let target = Target::from_ggml_id(ggml_type)
            .filter(|t| matches!(t, Target::Q4_0 | Target::Q4_1 | Target::Q5_0 | Target::Q5_1))
            .unwrap_or_else(|| panic!("{name}: ggml type {ggml_type} is not a legacy block type"));
        let elements: u64 = dims.iter().product();
        assert!(
            dims[0] % 32 == 0,
            "{name}: rows of {} elements do not tile 32-element legacy blocks",
            dims[0]
        );
        let data = target.encode(&Self::small_weights(elements, seed));
        self.tensor(name, ggml_type, dims, data)
    }

    /// The ORACLE twin of [`Self::legacy_tensor`]: the same weights through the
    /// same legacy encoder, then decoded with `compute`'s ggml-verified
    /// decoder and re-stored as Q8_0. A model built from these runs the
    /// already-trusted Q8_0 kernels on (to within Q8_0's own rounding) exactly
    /// the values the legacy kernels must reproduce, so the two installs'
    /// logits can be compared without a CPU forward pass.
    pub fn legacy_as_q8_0_tensor(self, name: &str, ggml_type: u32, dims: &[u64], seed: u8) -> Self {
        use crate::gguf_quantize::Target;
        if matches!(ggml_type, 34 | 35 | 39 | 40 | 41 | 42 | 64 | 65 | 66) {
            let elements: u64 = dims.iter().product();
            let n = elements as usize;
            let bytes =
                Self::ext_bytes(name, ggml_type, dims, &Self::small_weights(elements, seed));
            let decoded = match ggml_type {
                41 => compute::dequantize_q1_0(&bytes, n),
                35 => compute::dequantize_tq2_0(&bytes, n),
                39 => compute::dequantize_mxfp4(&bytes, n),
                42 => compute::dequantize_q2_0(&bytes, n),
                64 => compute::dequantize_iq1_xs(&bytes, n),
                65 => compute::dequantize_iq1_xxs(&bytes, n),
                66 => compute::dequantize_iq1_xxxs(&bytes, n),
                34 => compute::dequantize_tq1_0(&bytes, n),
                _ => compute::dequantize_nvfp4(&bytes, n),
            };
            return self.tensor(name, 8, dims, compute::quantize_q8_0(&decoded));
        }
        let target = Target::from_ggml_id(ggml_type)
            .filter(|t| matches!(t, Target::Q4_0 | Target::Q4_1 | Target::Q5_0 | Target::Q5_1))
            .unwrap_or_else(|| panic!("{name}: ggml type {ggml_type} is not a legacy block type"));
        let elements: u64 = dims.iter().product();
        let bytes = target.encode(&Self::small_weights(elements, seed));
        let n = elements as usize;
        let decoded = match target {
            Target::Q4_0 => compute::dequantize_q4_0(&bytes, n),
            Target::Q4_1 => compute::dequantize_q4_1(&bytes, n),
            Target::Q5_0 => compute::dequantize_q5_0(&bytes, n),
            _ => compute::dequantize_q5_1(&bytes, n),
        };
        self.tensor(name, 8, dims, compute::quantize_q8_0(&decoded))
    }

    /// Push a Q6_K tensor (256-element superblocks, 210 bytes each). The real
    /// Qwen file uses this for exactly one tensor, `output.weight`; a Gemma
    /// fixture has no such tensor because it ties its embeddings, so a
    /// K-quant fixture puts it on the attention projections instead.
    pub fn q6_k_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        assert!(
            dims[0] % 256 == 0,
            "{name}: rows of {} elements do not tile 256-element Q6_K superblocks",
            dims[0]
        );
        let data = compute::quantize_q6_k(&Self::small_weights(elements, seed));
        self.tensor(name, 14, dims, data)
    }

    /// Push a Q3_K tensor (256-element superblocks, 110 bytes each). The real
    /// Qwen2.5 Q3_K_M keeps its attention and FFN projections here. The
    /// fixture encoder writes a plain max-fit fit rather than ggml's
    /// `make_qx_quants` search, for the same reason the Q4_K helper's does:
    /// layout-exact, encoder-quality-simple.
    pub fn q3_k_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        assert!(
            dims[0] % 256 == 0,
            "{name}: rows of {} elements do not tile 256-element Q3_K superblocks",
            dims[0]
        );
        let data = compute::quantize_q3_k(&Self::small_weights(elements, seed));
        self.tensor(name, 11, dims, data)
    }

    /// Push an IQ tensor: IQ3_XXS (18), IQ4_NL (20) or IQ4_XS (23).
    ///
    /// UNLIKE EVERY OTHER TENSOR HELPER HERE, this does not quantize a weight
    /// vector, because this port has no IQ encoder and deliberately will not
    /// grow one (an encoder means a codebook nearest-neighbour search nothing
    /// calls, and the lossless-repack rule means real bytes arrive already
    /// quantized). It emits random VALID CODE POINTS instead, which is a
    /// stronger fixture rather than a weaker one: every byte of an IQ block
    /// is either a table index, a sign field or a scale field, all of whose
    /// bit patterns are legal, so random bytes cover the code space evenly
    /// where an encoder would only ever emit the subset it chooses.
    ///
    /// The scale `d` is per type and small on purpose. Each layout reaches a
    /// different maximum (IQ3_XXS's grid tops out at 62 under a scale nibble
    /// worth 7.75, IQ4_XS's table at 127 under a sub-scale of 32, IQ4_NL's at
    /// 127 flat), and an install whose weights reach +/-30 overflows FP16
    /// before the head - the trap `q8_0_tensor`'s `[-8, 7]` range exists for,
    /// arriving here by a different route.
    pub fn iq_tensor(self, name: &str, ggml_type: u32, dims: &[u64], seed: u8) -> Self {
        // (block elements, payload bytes after the f16 scale, f16 scale bits)
        let (elems, payload, d) = match ggml_type {
            18 => (256u64, 96usize, 0x1800u16), // IQ3_XXS, d = 2^-9
            20 => (32, 16, 0x2000),             // IQ4_NL,  d = 2^-7
            23 => (256, 134, 0x0C00),           // IQ4_XS,  d = 2^-12
            other => panic!("{name}: ggml type {other} is not an IQ layout"),
        };
        assert!(
            dims[0] % elems == 0,
            "{name}: rows of {} elements do not tile {elems}-element blocks",
            dims[0]
        );
        let total: u64 = dims.iter().product();
        let mut state = (seed as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let mut data = Vec::new();
        for _ in 0..total / elems {
            data.extend_from_slice(&d.to_le_bytes());
            for _ in 0..payload {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                data.push((state >> 33) as u8);
            }
        }
        self.tensor(name, ggml_type, dims, data)
    }

    /// Push an MXFP4 tensor (32-element blocks, 17 bytes each: one E8M0
    /// exponent then sixteen nibble-packed indices) -- ROADMAP M5's
    /// `gpt-oss`.
    ///
    /// Random valid code points like [`Self::iq_tensor`], and for the same
    /// reason: MXFP4 stores an INDEX into a fixed table, every nibble value
    /// is a legal index, and this port has no encoder. Its scale is not an
    /// f16 at all, so it cannot share that helper's two-byte header.
    ///
    /// `e = 124` is `2^-4` and the codebook tops out at 12, so the largest
    /// representable weight is 0.75. That is the same dynamic-range choice
    /// the two helpers above document, made in the one unit MXFP4 offers: an
    /// exponent. At `e = 128` the weights would reach 12 and a fixture
    /// install would overflow FP16 well before the head.
    pub fn mxfp4_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        assert!(
            dims[0] % 32 == 0,
            "{name}: rows of {} elements do not tile 32-element MXFP4 blocks",
            dims[0]
        );
        let mut state = (seed as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // `extend_from_slice` of a one-byte slice rather than `push`, only
        // because `clippy::same_item_push` reads a constant push in a loop as
        // a mistake. The block header really is one fixed byte per block.
        const EXPONENT: [u8; 1] = [124];
        let mut data = Vec::with_capacity((elements / 32 * 17) as usize);
        for _ in 0..elements / 32 {
            data.extend_from_slice(&EXPONENT);
            for _ in 0..16 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                data.push((state >> 33) as u8);
            }
        }
        self.tensor(name, 39, dims, data)
    }

    /// Push an F32 tensor whose every value is EXACTLY representable in
    /// BF16, which is what llama.cpp actually writes for norms: it upcasts
    /// tensors that were BF16 in the original checkpoint, so the low sixteen
    /// mantissa bits are all zero (measured, `gguf_f32_transcode_network.rs`).
    ///
    /// Zeros would satisfy that too, and used to be what this fixture wrote,
    /// but a zero tensor also narrows exactly under a BROKEN transcode. These
    /// values do not.
    pub fn f32_upcast_bf16_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        let mut data = Vec::with_capacity((elements * 4) as usize);
        for i in 0..elements {
            // Sign and exponent fixed around 1.0, mantissa varying, so every
            // value is an ordinary positive number a norm could hold.
            let bf16 = 0x3F00u16 | ((i as u16).wrapping_add(seed as u16) & 0x00FF);
            data.extend_from_slice(&((bf16 as u32) << 16).to_le_bytes());
        }
        self.tensor(name, 0, dims, data)
    }

    /// Push an F32 tensor of ordinary values, low mantissa bits included, so
    /// a quantizing transcode has real rounding to do. What the router is:
    /// GGUF ships it F32 and this port stores it INT8-affine.
    pub fn f32_tensor(self, name: &str, dims: &[u64], seed: u8) -> Self {
        let elements: u64 = dims.iter().product();
        let mut data = Vec::with_capacity((elements * 4) as usize);
        let mut s = (seed as u32).wrapping_mul(2_654_435_761).wrapping_add(17);
        for _ in 0..elements {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0;
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.tensor(name, 0, dims, data)
    }
}
