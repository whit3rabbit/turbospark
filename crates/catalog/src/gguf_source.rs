//! Shared header-only split discovery for probe and installation.
use crate::hf::{Client, RepoRef};
use repack::{ByteProgressCallback, CancelFlag, GgufSet, HttpRangeSource};

pub(crate) fn load(
    client: &Client,
    repo: &RepoRef,
    file: &str,
    progress: Option<&ByteProgressCallback>,
    cancel: Option<&CancelFlag>,
) -> Result<GgufSet<HttpRangeSource>, String> {
    let first = crate::stream::range_source(repo.file_url(file), progress, client, cancel);
    let header = repack::fetch_gguf_header(&first).map_err(|e| format!("{file}: {e}"))?;
    let names = repack::gguf_shard_names(file, &header)?;
    let mut shards = Vec::with_capacity(names.len());
    let mut initial = Some((header, first));
    for name in &names {
        if cancel.is_some_and(|flag| flag.is_cancelled()) {
            return Err(crate::install::INSTALL_CANCELLED.into());
        }
        let url = repo.file_url(name);
        let (header, source) = match initial.take() {
            Some(pair) => pair,
            None => {
                let source = crate::stream::range_source(url.clone(), progress, client, cancel);
                let header =
                    repack::fetch_gguf_header(&source).map_err(|e| format!("{name}: {e}"))?;
                (header, source)
            }
        };
        let length = client
            .content_length(&url)?
            .ok_or_else(|| format!("{name}: missing content length"))?;
        shards.push((header, source, length));
    }
    GgufSet::new(shards)
}
