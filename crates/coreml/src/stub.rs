//! Non-macOS build: same API, every entry point refuses. Keeps the crate
//! (and anything that names its types) checkable for portable targets.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{CoreMlError, Result};
use crate::{Capability, FeatureSpec, LoadOptions, Tensor};

pub fn supports(_cap: Capability) -> bool {
    false
}

pub fn os_version() -> (u32, u32, u32) {
    (0, 0, 0)
}

pub fn compile_if_needed(_path: &Path, _cache_root: &Path) -> Result<PathBuf> {
    Err(CoreMlError::UnsupportedPlatform)
}

#[derive(Debug)]
pub struct CoreMlModel {
    _private: (),
}

impl CoreMlModel {
    pub fn load(_path: &Path, _opts: &LoadOptions) -> Result<Self> {
        Err(CoreMlError::UnsupportedPlatform)
    }

    pub fn open(_path: &Path, _cache_root: &Path, _opts: &LoadOptions) -> Result<Self> {
        Err(CoreMlError::UnsupportedPlatform)
    }

    pub fn inputs(&self) -> &[FeatureSpec] {
        &[]
    }

    pub fn outputs(&self) -> &[FeatureSpec] {
        &[]
    }

    pub fn states(&self) -> &[FeatureSpec] {
        &[]
    }

    pub fn path(&self) -> &Path {
        Path::new("")
    }

    pub fn function(&self) -> Option<&str> {
        None
    }

    pub fn predict(&self, _inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>> {
        Err(CoreMlError::UnsupportedPlatform)
    }

    pub fn new_state(&self) -> Result<CoreMlState> {
        Err(CoreMlError::UnsupportedPlatform)
    }

    pub fn predict_with_state(
        &self,
        _state: &mut CoreMlState,
        _inputs: &[(&str, &Tensor)],
    ) -> Result<HashMap<String, Tensor>> {
        Err(CoreMlError::UnsupportedPlatform)
    }
}

#[derive(Debug)]
pub struct CoreMlState {
    _private: (),
}

impl CoreMlState {
    pub fn specs(&self) -> &[FeatureSpec] {
        &[]
    }

    pub fn copy_from(&mut self, _src: &CoreMlState) -> Result<()> {
        Err(CoreMlError::UnsupportedPlatform)
    }

    pub fn read(&self, _name: &str) -> Result<Tensor> {
        Err(CoreMlError::UnsupportedPlatform)
    }
}
