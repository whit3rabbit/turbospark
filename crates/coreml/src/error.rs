use std::fmt;

/// Every failure the runner reports. Variants are coarse on purpose: callers
/// either refuse the model (platform, OS, load, compile) or report a bad call
/// (shape, dtype, missing feature), and the message carries the detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreMlError {
    /// Built for a target without Core ML (anything but macOS).
    UnsupportedPlatform,
    /// The running OS lacks an API the model needs. `required` is the macOS
    /// version that introduced it.
    OsTooOld {
        feature: &'static str,
        required: &'static str,
    },
    Load(String),
    Compile(String),
    Shape {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    DType {
        name: String,
        expected: String,
        got: String,
    },
    MissingFeature(String),
    /// A value would not survive the f32 -> f16 conversion (overflow or a
    /// non-finite input). Core ML would otherwise compute on inf silently.
    NonFinite {
        name: String,
        index: usize,
    },
    Prediction(String),
    /// The Objective-C runtime raised an exception instead of returning an
    /// NSError. Caught so it cannot abort the process.
    Exception(String),
    Io(String),
}

impl fmt::Display for CoreMlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(f, "Core ML is only available on macOS"),
            Self::OsTooOld { feature, required } => {
                write!(f, "{feature} requires macOS {required} or newer")
            }
            Self::Load(m) => write!(f, "Core ML model load failed: {m}"),
            Self::Compile(m) => write!(f, "Core ML model compile failed: {m}"),
            Self::Shape {
                name,
                expected,
                got,
            } => write!(
                f,
                "feature {name}: expected shape {expected:?}, got {got:?}"
            ),
            Self::DType {
                name,
                expected,
                got,
            } => write!(f, "feature {name}: expected dtype {expected}, got {got}"),
            Self::MissingFeature(n) => write!(f, "feature {n} is missing"),
            Self::NonFinite { name, index } => write!(
                f,
                "feature {name}: element {index} is not finite after f16 conversion"
            ),
            Self::Prediction(m) => write!(f, "Core ML prediction failed: {m}"),
            Self::Exception(m) => write!(f, "Core ML raised an exception: {m}"),
            Self::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for CoreMlError {}

impl From<std::io::Error> for CoreMlError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, CoreMlError>;
