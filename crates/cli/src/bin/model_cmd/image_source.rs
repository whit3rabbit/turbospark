//! Materialize the small Diffusers source tree needed by the image packer.

use std::path::Path;
use std::sync::Arc;

use catalog::{Client, RepoRef};

/// Download a pinned image repository into a temporary source tree.
///
/// File selection, ranged resumable downloads, and per-family normalization
/// live in `catalog::materialize_image_source`, the same function the app's
/// install path uses. This used to be a second copy of the selection rules,
/// which silently lagged the catalog: it had no `processor/` layout rule, so
/// every Qwen-Image row failed to pack from the CLI, and it dropped the
/// license files a Qwen install must keep.
pub(super) fn materialize(
    client: &Client,
    repo: &RepoRef,
    destination: &Path,
) -> Result<u64, String> {
    if destination.exists() {
        return Err(format!(
            "refusing to overwrite image source staging directory {}",
            destination.display()
        ));
    }
    catalog::materialize_image_source(client, repo, destination, None, |_| {}, Arc::new(|_, _| {}))
}
