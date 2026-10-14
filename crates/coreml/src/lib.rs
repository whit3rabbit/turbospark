//! Apple Core ML runner.
//!
//! Loads compiled (`.mlmodelc`) or packaged (`.mlpackage`, `.mlmodel`)
//! models, checks inputs against the model description, marshals dense
//! row-major tensors to and from `MLMultiArray`, and runs stateless or
//! stateful (`MLState`) predictions. Model-specific pipelines (Parakeet,
//! EmbeddingGemma2, Unlimited OCR) live in `turbospark-runtime`; this crate
//! knows nothing about any model.
//!
//! On targets other than macOS every entry point returns
//! [`CoreMlError::UnsupportedPlatform`], so the crate stays in the portable
//! cross-target check. All unsafe code is in `sys` (see `AGENTS.md`).

#![deny(unsafe_code)]
// The marshalling helpers are only reachable through `sys` on macOS.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

mod cache;
mod error;
mod tensor;

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod sys;

#[cfg(not(target_os = "macos"))]
mod stub;

pub use cache::{cache_key, compiled_path_for};
pub use error::{CoreMlError, Result};
pub use tensor::{element_count, DType, Tensor, TensorData};

#[cfg(target_os = "macos")]
pub use sys::{compile_if_needed, os_version, supports, CoreMlModel, CoreMlState};

#[cfg(not(target_os = "macos"))]
pub use stub::{compile_if_needed, os_version, supports, CoreMlModel, CoreMlState};

/// Which processors Core ML may schedule on. Mirrors `MLComputeUnits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComputeUnits {
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
    #[default]
    All,
}

impl ComputeUnits {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuOnly => "cpu",
            Self::CpuAndGpu => "cpu_gpu",
            Self::CpuAndNeuralEngine => "cpu_ane",
            Self::All => "all",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "cpu" => Some(Self::CpuOnly),
            "cpu_gpu" | "gpu" => Some(Self::CpuAndGpu),
            "cpu_ane" | "ane" => Some(Self::CpuAndNeuralEngine),
            "all" => Some(Self::All),
            _ => None,
        }
    }
}

/// Runtime capabilities that arrived after the oldest macOS this crate is
/// linked into (the Swift package targets macOS 13). objc2 resolves classes
/// and selectors at runtime, so calling one of these on an older OS would
/// panic or raise; [`supports`] is checked first and the caller gets
/// [`CoreMlError::OsTooOld`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// `MLModelConfiguration.functionName`, for multifunction packages.
    MultiFunction,
    /// `MLState` and `predictionFromFeatures:usingState:`.
    State,
    /// `compileModelAtURL:completionHandler:` (14.4). The synchronous
    /// variant is the fallback before that.
    AsyncCompile,
}

impl Capability {
    pub fn required_macos(self) -> &'static str {
        match self {
            Self::MultiFunction | Self::State => "15.0",
            Self::AsyncCompile => "14.4",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::MultiFunction => "multifunction Core ML models",
            Self::State => "stateful Core ML models",
            Self::AsyncCompile => "asynchronous Core ML compilation",
        }
    }

    pub(crate) fn too_old(self) -> CoreMlError {
        CoreMlError::OsTooOld {
            feature: self.name(),
            required: self.required_macos(),
        }
    }
}

/// How to load one model (or one function of a multifunction package).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadOptions {
    pub compute_units: ComputeUnits,
    /// Function to bind in a multifunction package. `None` loads the
    /// package's default function. Requires [`Capability::MultiFunction`].
    pub function: Option<String>,
}

impl LoadOptions {
    pub fn new(compute_units: ComputeUnits) -> Self {
        Self {
            compute_units,
            function: None,
        }
    }

    pub fn with_function(mut self, name: impl Into<String>) -> Self {
        self.function = Some(name.into());
        self
    }
}

/// What a feature in the model description declares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureKind {
    /// A multi-array. `shape` is `Some` only when the shape is fixed; a
    /// flexible (enumerated or range) shape is left to Core ML to validate.
    /// `dtype` is `None` for element types this crate does not marshal.
    MultiArray {
        dtype: Option<DType>,
        shape: Option<Vec<usize>>,
    },
    /// A state buffer of a stateful model.
    State {
        dtype: Option<DType>,
        shape: Vec<usize>,
    },
    /// Any other feature type (image, string, dictionary, ...), by its
    /// `MLFeatureType` raw value. Not marshalled.
    Other(isize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSpec {
    pub name: String,
    pub kind: FeatureKind,
    pub optional: bool,
}

/// Checks one input tensor against its declared feature. Pure, so the
/// refusal paths are testable without a model.
pub fn check_input(spec: &FeatureSpec, tensor: &Tensor) -> Result<()> {
    match &spec.kind {
        FeatureKind::MultiArray { dtype, shape } => {
            match dtype {
                Some(d) if *d == tensor.dtype() => {}
                Some(d) => {
                    return Err(CoreMlError::DType {
                        name: spec.name.clone(),
                        expected: d.to_string(),
                        got: tensor.dtype().to_string(),
                    })
                }
                None => {
                    return Err(CoreMlError::DType {
                        name: spec.name.clone(),
                        expected: "a supported multi-array type".into(),
                        got: tensor.dtype().to_string(),
                    })
                }
            }
            if let Some(s) = shape {
                if s.as_slice() != tensor.shape() {
                    return Err(CoreMlError::Shape {
                        name: spec.name.clone(),
                        expected: s.clone(),
                        got: tensor.shape().to_vec(),
                    });
                }
            }
            Ok(())
        }
        FeatureKind::State { .. } => Err(CoreMlError::Prediction(format!(
            "{} is a state buffer, not an input",
            spec.name
        ))),
        FeatureKind::Other(t) => Err(CoreMlError::DType {
            name: spec.name.clone(),
            expected: format!("MLFeatureType {t}"),
            got: tensor.dtype().to_string(),
        }),
    }
}

/// Checks a full input set: every name must be declared, every required
/// input supplied, and each tensor must match.
pub fn check_inputs(specs: &[FeatureSpec], inputs: &[(&str, &Tensor)]) -> Result<()> {
    for (name, t) in inputs {
        let spec = specs
            .iter()
            .find(|s| s.name == *name)
            .ok_or_else(|| CoreMlError::MissingFeature(format!("{name} (not a model input)")))?;
        check_input(spec, t)?;
    }
    for s in specs {
        if !s.optional && !inputs.iter().any(|(n, _)| *n == s.name) {
            return Err(CoreMlError::MissingFeature(s.name.clone()));
        }
    }
    let mut names: Vec<&str> = inputs.iter().map(|(n, _)| *n).collect();
    names.sort_unstable();
    if names.windows(2).any(|w| w[0] == w[1]) {
        return Err(CoreMlError::Prediction("duplicate input name".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, dtype: DType, shape: Option<Vec<usize>>) -> FeatureSpec {
        FeatureSpec {
            name: name.into(),
            kind: FeatureKind::MultiArray {
                dtype: Some(dtype),
                shape,
            },
            optional: false,
        }
    }

    #[test]
    fn check_inputs_refuses_wrong_dtype_shape_unknown_and_missing() {
        let specs = vec![
            spec("x", DType::F32, Some(vec![1, 4])),
            spec("n", DType::I32, None),
        ];
        let x = Tensor::f32(vec![1, 4], vec![0.0; 4]).unwrap();
        let n = Tensor::i32(vec![1], vec![3]).unwrap();
        assert!(check_inputs(&specs, &[("x", &x), ("n", &n)]).is_ok());

        let bad_shape = Tensor::f32(vec![4], vec![0.0; 4]).unwrap();
        assert!(matches!(
            check_inputs(&specs, &[("x", &bad_shape), ("n", &n)]),
            Err(CoreMlError::Shape { .. })
        ));
        let bad_dtype = Tensor::i32(vec![1, 4], vec![0; 4]).unwrap();
        assert!(matches!(
            check_inputs(&specs, &[("x", &bad_dtype), ("n", &n)]),
            Err(CoreMlError::DType { .. })
        ));
        assert!(matches!(
            check_inputs(&specs, &[("x", &x)]),
            Err(CoreMlError::MissingFeature(m)) if m == "n"
        ));
        assert!(matches!(
            check_inputs(&specs, &[("x", &x), ("n", &n), ("zzz", &n)]),
            Err(CoreMlError::MissingFeature(_))
        ));
    }

    #[test]
    fn flexible_shape_is_not_checked_here() {
        let specs = vec![spec("x", DType::F16, None)];
        let x = Tensor::f16_from_f32("x", vec![3, 7], &[0.0; 21]).unwrap();
        assert!(check_inputs(&specs, &[("x", &x)]).is_ok());
    }

    #[test]
    fn compute_units_round_trip() {
        for u in [
            ComputeUnits::CpuOnly,
            ComputeUnits::CpuAndGpu,
            ComputeUnits::CpuAndNeuralEngine,
            ComputeUnits::All,
        ] {
            assert_eq!(ComputeUnits::parse(u.as_str()), Some(u));
        }
        assert_eq!(ComputeUnits::parse("tpu"), None);
    }
}
