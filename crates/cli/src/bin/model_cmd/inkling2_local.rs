//! `pull-local`: installs Inkling 2 Mini from files on disk, a LookAtMyQuant
//! packed release directory or an Inkling 2 GGUF. The intake, verification
//! and recording are `catalog::install_inkling2_local`, shared with the FFI;
//! this wrapper resolves the destination the way `pull` does and prints
//! progress. `pull --repo` of such a repository lands in [`pull_repo`],
//! which downloads the files first (`catalog::install_inkling2_repo`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use catalog::{Client, InstallKind, InstallTarget, Store};

use super::destination::{self, Destination};
use super::{human_bytes, text_destination};
use crate::{Error, Options};

/// `pull-local <PATH> --alias NAME [--tokenizer-from DIR] [--out DIR | --library L]`.
pub fn pull_local(store: &Store, positionals: &[String], options: &Options) -> Result<(), Error> {
    let [path] = positionals else {
        return Err(Error::Usage(
            "pull-local takes one <PATH>: a LookAtMyQuant Inkling release directory or an \
             Inkling 2 GGUF"
                .to_string(),
        ));
    };
    let alias = options
        .alias
        .as_deref()
        .ok_or_else(|| Error::Usage("pull-local needs --alias <NAME>".to_string()))?;
    let path = PathBuf::from(path);
    let source = catalog::detect_inkling2_local(&path)
        .map_err(Error::Failed)?
        .ok_or_else(|| {
            Error::Failed(format!(
                "{} is neither a LookAtMyQuant Inkling release directory (manifest.json \
                 format {:?}) nor an Inkling 2 GGUF; for other models use pull --repo",
                path.display(),
                repack::LAMQ_INKLING2_FORMAT
            ))
        })?;
    let tokenizer_from = options.tokenizer_from.as_deref().map(Path::new);
    if matches!(source, catalog::Inkling2LocalSource::Gguf(_)) && tokenizer_from.is_none() {
        return Err(Error::Usage(
            "a GGUF carries no tokenizer.json: pass --tokenizer-from <DIR>, a directory holding \
             the Inkling 2 tokenizer files (an HF install, or a download of \
             thinkingmachines/Inkling-2-Mini)"
                .to_string(),
        ));
    }
    let destination = text_destination(store, InstallKind::Text, alias, options)?;
    let (target, env) = target_of(store, alias, destination)?;
    let available = model_io::available_bytes(target.dir.parent().unwrap_or(&target.dir))
        .map_err(Error::Failed)?;
    let started = Instant::now();
    let origin = path
        .canonicalize()
        .unwrap_or_else(|_| path.clone())
        .display()
        .to_string();
    let installed = catalog::install_inkling2_local(
        &source,
        alias,
        &target,
        tokenizer_from,
        &format!("local:{origin}"),
        "local",
        available,
        |line| eprintln!("[{:7.1}s] {line}", started.elapsed().as_secs_f64()),
        store,
        &env,
    )
    .map_err(destination::failure)?;
    println!(
        "installed {} ({} Inkling 2, {}) at {}",
        installed.model.alias,
        source.kind(),
        human_bytes(installed.model.install_bytes),
        installed.model.path.display()
    );
    Ok(())
}

/// The install target for a resolved destination. `--out` is a This Mac row
/// whose folder is the one named.
fn target_of(
    store: &Store,
    alias: &str,
    destination: Destination,
) -> Result<(InstallTarget, catalog::library::LocationEnv), Error> {
    match destination {
        Destination::Target(target, env) => Ok((target, env)),
        Destination::Out(dir) => {
            let env = catalog::library::LocationEnv::system(store.root()).map_err(Error::Failed)?;
            let mut target =
                catalog::install_destination(store, None, InstallKind::Text, alias, &env)
                    .map_err(destination::failure)?;
            target.dir = dir;
            Ok((target, env))
        }
    }
}

/// `pull --repo` of an Inkling 2 GGUF or LookAtMyQuant release: download
/// the files whole, then the local intake.
pub(super) fn pull_repo(
    store: &Store,
    client: &Client,
    source: &catalog::Inkling2RepoSource,
    alias: &str,
    destination: Destination,
) -> Result<(), Error> {
    let (target, env) = target_of(store, alias, destination)?;
    let pb = super::progress::byte_progress_bar(source.download_bytes());
    let pb_msg = pb.clone();
    let pb_bytes = pb.clone();
    let started = Instant::now();
    let installed = catalog::install_inkling2_repo(
        source,
        alias,
        &target,
        client,
        move |line| pb_msg.println(format!("[{:7.1}s] {line}", started.elapsed().as_secs_f64())),
        &mut move |n| pb_bytes.inc(n),
        None,
        store,
        &env,
    )
    .inspect_err(|_| pb.finish_and_clear())
    .map_err(destination::failure)?;
    pb.finish_and_clear();
    println!(
        "installed {} ({} Inkling 2 from {}, {}) at {}",
        installed.model.alias,
        match source.kind {
            catalog::Inkling2RepoKind::Lamq => "lamq",
            catalog::Inkling2RepoKind::Gguf => "gguf",
        },
        source.repo,
        human_bytes(installed.model.install_bytes),
        installed.model.path.display()
    );
    Ok(())
}
