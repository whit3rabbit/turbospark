//! The model-management surface: browse, probe, install.
//!
//! Portable. Nothing here decodes, so a GUI can list the catalog and install
//! a model on a platform where it could not then run one -- which is the
//! honest behaviour, since the artifact is the same either way.
//!
//! The catalog and store rows are `crates/catalog`'s own `Serialize` types
//! passed straight through, so a GUI's rows and `turbospark-model list`'s
//! rows are the same data and cannot drift. `ProbeReport` is NOT
//! `Serialize` (it carries an `ArchConfig` and a `ModelFamily`), so its
//! projection below is written by hand -- deliberately narrow, since a GUI
//! needs the verdict and the arithmetic behind it rather than the whole
//! config.

mod catalog;
mod control_vector;
mod fit;
mod image_install;
mod install;
mod probe;

pub(crate) use catalog::{
    catalog_json, delete, delete_image, image_installed_json, install_bytes, installed_json,
};
pub use control_vector::control_vector_info_json;
pub(crate) use fit::{context_ladder_json, recommend_json};
pub(crate) use image_install::{catalog_json as image_catalog_json, install as image_install};
pub(crate) use install::{cancel_active_installs, install, install_repo, installs_finished};
pub use install::{TS_INSTALL_BYTES, TS_INSTALL_STAGE};
pub(crate) use probe::{probe_json, repo_variants_json};
