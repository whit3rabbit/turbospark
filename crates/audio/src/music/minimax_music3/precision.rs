//! Logical Music 3 activation precision. Storage remains portable f32.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DType {
    #[default]
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Music3Precision {
    #[default]
    Checkpoint,
    Float32,
}

impl DType {
    pub fn from_checkpoint(dtype: &str) -> crate::Result<Self> {
        match dtype {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            _ => Err(crate::SpeechError::Unsupported {
                why: format!("unsupported Music 3 floating dtype {dtype}"),
            }),
        }
    }
    pub fn promote(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            Self::F32
        }
    }
    pub fn round(self, value: f32) -> f32 {
        if self == Self::F32 {
            return value;
        }
        let bits = value.to_bits();
        if self == Self::Bf16 {
            if bits & 0x7fff_ffff > 0x7f80_0000 {
                return f32::from_bits((bits & 0xffff_0000) | 0x0040_0000);
            }
            return f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000);
        }
        if !value.is_finite() || value == 0.0 {
            return value;
        }
        let magnitude = value.abs();
        if magnitude >= 65_520.0 {
            return value.signum() * f32::INFINITY;
        }
        // Every binary16 value is an exact f32 value; round its significand
        // directly, including subnormal ties, without a platform ABI dependency.
        let exponent = ((bits >> 23) & 0xff) as i32 - 127;
        let step = if exponent < -14 {
            f32::from_bits(103 << 23)
        } else {
            2.0f32.powi(exponent - 10)
        };
        let scaled = magnitude / step;
        let floor = scaled.floor();
        let fraction = scaled - floor;
        let rounded = if fraction > 0.5 || (fraction == 0.5 && (floor as u32 & 1) != 0) {
            floor + 1.0
        } else {
            floor
        };
        (rounded * step).copysign(value)
    }
    pub fn round_slice(self, values: &mut [f32]) {
        if self != Self::F32 {
            for value in values {
                *value = self.round(*value);
            }
        }
    }
    pub(crate) fn sigmoid(self, value: f32) -> f32 {
        if self == Self::F32 {
            return 1.0 / (1.0 + (-value).exp());
        }
        // The pinned Metal unary uses a symmetric native-typed expression.
        let exponential = self.round(value.abs().exp());
        let y = self.round(1.0 / self.round(1.0 + exponential));
        if value < 0.0 {
            y
        } else {
            self.round(1.0 - y)
        }
    }
    pub(crate) fn silu(self, value: f32) -> f32 {
        if self == Self::F32 {
            value / (1.0 + (-value).exp())
        } else {
            self.round(value * self.sigmoid(value))
        }
    }
}

impl Music3Precision {
    pub(crate) fn dtype(self, dtype: DType) -> DType {
        if self == Self::Float32 {
            DType::F32
        } else {
            dtype
        }
    }
}

pub(crate) fn rms_norm(
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    dtype: DType,
) -> Vec<f32> {
    let mut out = x.to_vec();
    if dtype == DType::F32 {
        crate::ops::rmsnorm(&mut out, rows, cols, w, eps);
        return out;
    }
    for row in out.chunks_exact_mut(cols) {
        let variance = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let scale = 1.0 / (variance + eps).sqrt();
        for (value, weight) in row.iter_mut().zip(w) {
            *value = dtype.round(dtype.round(*value * scale) * weight);
        }
    }
    out
}

pub(crate) fn layer_norm(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
    dtype: DType,
) -> Vec<f32> {
    let mut out = x.to_vec();
    if dtype == DType::F32 {
        crate::ops::layernorm(&mut out, rows, cols, w, bias, eps);
        return out;
    }
    for row in out.chunks_exact_mut(cols) {
        let mean = row.iter().sum::<f32>() / cols as f32;
        let variance = row.iter().map(|v| v * v).sum::<f32>() / cols as f32 - mean * mean;
        let scale = 1.0 / (variance + eps).sqrt();
        for (i, value) in row.iter_mut().enumerate() {
            let normalized = dtype.round((*value - mean) * scale);
            *value = dtype.round(normalized * w[i] + bias.map_or(0.0, |b| b[i]));
        }
    }
    out
}

pub(crate) fn fourier(
    timesteps: &[f32],
    weight: &[f32],
    input_dtype: DType,
    weight_dtype: DType,
) -> Vec<f32> {
    let dtype = input_dtype.promote(weight_dtype);
    let two_pi = input_dtype.round((2.0 * std::f64::consts::PI) as f32);
    let mut out = Vec::with_capacity(timesteps.len() * weight.len() * 2);
    for &timestep in timesteps {
        let angle = input_dtype.round(two_pi * input_dtype.round(timestep));
        let angles: Vec<f32> = weight.iter().map(|w| dtype.round(angle * w)).collect();
        out.extend(angles.iter().map(|a| dtype.round(a.cos())));
        out.extend(angles.iter().map(|a| dtype.round(a.sin())));
    }
    out
}

pub(crate) fn sum_rows(input: &[f32], rows: usize, cols: usize, dtype: DType) -> Vec<f32> {
    let mut out = vec![0.0; cols];
    for row in input.chunks_exact(cols).take(rows) {
        for (v, add) in out.iter_mut().zip(row) {
            *v = dtype.round(*v + add);
        }
    }
    dtype.round_slice(&mut out);
    out
}

#[cfg(test)]
// Keep independently recorded values and exact binary tie boundaries readable.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    #[test]
    fn native_rms_norm_rounds_before_learned_weight() {
        // Independent mx.fast.rms_norm values from pinned MLX 0.32.3.
        let x = [0.984375, 2.46875, -0.70703125, 0.2001953125];
        let w = [1.203125, 0.90234375, 1.796875, 0.69921875];
        assert_eq!(
            rms_norm(&x, &w, 1, 4, 1e-6, DType::Bf16),
            vec![0.859375, 1.6171875, -0.91796875, 0.1015625]
        );
    }
    fn values(case: &serde_json::Value, key: &str) -> Vec<f32> {
        case[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
    fn size(case: &serde_json::Value, key: &str) -> usize {
        case[key].as_u64().unwrap() as usize
    }

    #[test]
    fn portable_native_operations_match_pinned_mlx_fixtures() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../testdata/minimax_music3/precision/ops.json"
        ))
        .unwrap();
        let mut failures = Vec::new();
        let mut checked = 0;
        for case in fixture["cases"].as_array().unwrap() {
            let dtype = DType::Bf16;
            let actual = match case["op"].as_str().unwrap() {
                "rms_norm" => rms_norm(
                    &values(case, "input"),
                    &values(case, "weight"),
                    size(case, "rows"),
                    size(case, "cols"),
                    case["eps"].as_f64().unwrap() as f32,
                    dtype,
                ),
                "layer_norm" => layer_norm(
                    &values(case, "input"),
                    &values(case, "weight"),
                    Some(&values(case, "bias")),
                    size(case, "rows"),
                    size(case, "cols"),
                    case["eps"].as_f64().unwrap() as f32,
                    dtype,
                ),
                "sigmoid" => values(case, "input")
                    .iter()
                    .map(|&x| dtype.sigmoid(x))
                    .collect(),
                "silu" => values(case, "input")
                    .iter()
                    .map(|&x| dtype.silu(x))
                    .collect(),
                "swiglu" => values(case, "input")
                    .iter()
                    .zip(values(case, "up"))
                    .map(|(&x, u)| dtype.round(dtype.silu(x) * u))
                    .collect(),
                "scalar_multiply" => {
                    let scalar = dtype.round(case["scalar"].as_f64().unwrap() as f32);
                    values(case, "input")
                        .iter()
                        .map(|&x| dtype.round(x * scalar))
                        .collect()
                }
                "multiply" => {
                    let promoted = dtype.promote(DType::F32);
                    values(case, "input")
                        .iter()
                        .zip(values(case, "operand"))
                        .map(|(&x, o)| promoted.round(x * o))
                        .collect()
                }
                "add" => values(case, "input")
                    .iter()
                    .zip(values(case, "operand"))
                    .map(|(&x, o)| dtype.round(x + o))
                    .collect(),
                "sum" => sum_rows(
                    &values(case, "input"),
                    case["input_shape"][0].as_u64().unwrap() as usize,
                    case["input_shape"][1].as_u64().unwrap() as usize,
                    dtype,
                ),
                // Metal pow has device-specific rounding at the supplied
                // native_power ties; those fixtures exercise the device seam.
                "snake" if case.get("native_power").is_some() => continue,
                "snake" => {
                    let mut x = values(case, "input");
                    super::super::vocoder::snake(
                        &mut x,
                        &values(case, "alpha"),
                        case["input_shape"][1].as_u64().unwrap() as usize,
                        case["input_shape"][2].as_u64().unwrap() as usize,
                        dtype,
                    );
                    x
                }
                "euler_guidance" => values(case, "unconditional")
                    .iter()
                    .zip(values(case, "conditional"))
                    .map(|(&u, c)| {
                        super::super::euler::guided_velocity(
                            u,
                            c,
                            case["guidance"].as_f64().unwrap() as f32,
                            dtype,
                        )
                    })
                    .collect(),
                "euler_update" => values(case, "input")
                    .iter()
                    .zip(values(case, "velocity"))
                    .map(|(&x, v)| {
                        super::super::euler::update(
                            x,
                            v,
                            case["delta"].as_f64().unwrap() as f32,
                            dtype,
                            dtype,
                        )
                    })
                    .collect(),
                "overlap_blend" => values(case, "noise")
                    .iter()
                    .zip(values(case, "previous"))
                    .map(|(&x, p)| {
                        super::super::euler::overlap_blend(
                            x,
                            p,
                            case["sigma"].as_f64().unwrap() as f32,
                            dtype,
                        )
                    })
                    .collect(),
                "fourier" => fourier(
                    &values(case, "timestep"),
                    &values(case, "weight"),
                    dtype,
                    dtype,
                ),
                "normal_from_uniform" => {
                    super::super::rng::normal_from_uniform(&values(case, "input"), dtype)
                }
                "partial_rope" => {
                    let (batch, seq, heads, dim, rotary) = (
                        size(case, "batch"),
                        size(case, "seq"),
                        size(case, "heads"),
                        size(case, "dim"),
                        size(case, "rotary_dim"),
                    );
                    let input = values(case, "input");
                    let cos = values(case, "cos");
                    let sin = values(case, "sin");
                    let cos: Vec<f32> = (0..seq)
                        .flat_map(|t| cos[t * rotary..t * rotary + rotary / 2].to_vec())
                        .collect();
                    let sin: Vec<f32> = (0..seq)
                        .flat_map(|t| sin[t * rotary..t * rotary + rotary / 2].to_vec())
                        .collect();
                    let mut output = input.clone();
                    for b in 0..batch {
                        for h in 0..heads {
                            let mut plane: Vec<f32> = (0..seq)
                                .flat_map(|t| {
                                    input[((b * seq + t) * heads + h) * dim
                                        ..((b * seq + t) * heads + h + 1) * dim]
                                        .to_vec()
                                })
                                .collect();
                            super::super::dit::partial_rope_typed(
                                &mut plane, dim, rotary, &cos, &sin, dtype,
                            );
                            for t in 0..seq {
                                let offset = ((b * seq + t) * heads + h) * dim;
                                output[offset..offset + dim]
                                    .copy_from_slice(&plane[t * dim..(t + 1) * dim]);
                            }
                        }
                    }
                    output
                }
                _ => continue,
            };
            checked += 1;
            let expected = values(case, "expected");
            assert_eq!(actual.len(), expected.len(), "{} shape", case["name"]);
            let mismatches: Vec<usize> = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .filter_map(|(i, (a, b))| (a != b).then_some(i))
                .collect();
            if let Some(&i) = mismatches.first() {
                failures.push(format!(
                    "{}: {} unequal, first [{i}] {} != {}",
                    case["name"],
                    mismatches.len(),
                    actual[i],
                    expected[i]
                ));
            }
        }
        assert!(checked >= 18, "missing portable fixture coverage");
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn binary16_and_bfloat16_round_ties_to_even() {
        assert_eq!(DType::Bf16.round(1.00390625), 1.0);
        assert_eq!(DType::Bf16.round(1.01171875), 1.015625);
        assert_eq!(DType::F16.round(1.00048828125), 1.0);
        assert_eq!(DType::F16.round(1.00146484375), 1.001953125);
        assert_eq!(DType::F16.round(2.0f32.powi(-25)), 0.0);
        assert_eq!(DType::F16.round(3.0 * 2.0f32.powi(-25)), 2.0f32.powi(-23));
        assert_eq!(DType::F16.round(65520.0), f32::INFINITY);
        assert!(DType::Bf16.round(f32::NAN).is_nan());
        assert_eq!(DType::F16.promote(DType::Bf16), DType::F32);
    }

    #[test]
    fn native_rotary_tables_keep_positive_power_reciprocal_boundary() {
        let (cos, sin) = super::super::backend::rotary_tables(3, 4, 7.0);
        // Independent MLX 0.32.3 half tables. The device backend supplies
        // Metal's exact transcendental bits; portable libm can differ by ulps.
        let expected_cos = [
            1.0,
            1.0,
            0.5403022766113281,
            0.9294176697731018,
            -0.416146844625473,
            0.727634608745575,
        ];
        let expected_sin = [
            0.0,
            0.0,
            0.8414710164070129,
            0.36902937293052673,
            0.9092974066734314,
            0.6859648823738098,
        ];
        for (actual, expected) in cos
            .iter()
            .chain(&sin)
            .zip(expected_cos.into_iter().chain(expected_sin))
        {
            assert!((actual - expected).abs() <= 2.0 * f32::EPSILON);
        }
        let (_, legacy) = crate::ops::rope_tables(3, 4, 7.0);
        assert_ne!(
            sin, legacy,
            "native tables must retain the reciprocal operation"
        );
    }
}
