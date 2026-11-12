//! The audio commands: `list-audio`, `pull-audio`, and the audio side of
//! `path` and `rm`.
//!
//! Every pinned row in the audio catalog installs here through the same
//! `AudioCatalog::install_acknowledged` the app calls over the FFI, so a CLI
//! pull lands in `<root>/models/audio/<alias>.gturbo` with the receipt the app
//! verifies before it loads anything. MiniMax Music 3 keeps its own installer
//! (`install_music`), which promotes a different source layout. Fish Audio
//! Drama 3 converts its files on install and has its own command.
//!
//! Some rows run only in the TurboSpark app (FluidAudio's Core ML models).
//! They install here all the same: the CLI is how a headless box or a script
//! gets the bytes in place, and the listing says which rows the Rust side can
//! run.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use catalog::{AudioCatalog, AudioProfile, Catalog, Client, InstallKind, Store};

use super::{destination, human_bytes, install_sigint_handler, progress, INTERRUPTED};
use crate::{Error, Options};

/// What the listing says about where a row runs.
fn runs_where(profile: &AudioProfile) -> &'static str {
    let caps = &profile.capabilities;
    if caps.backend == "swift_coreml" {
        "app only"
    } else if caps.can_run {
        "rust"
    } else {
        "no runner"
    }
}

/// Every pinned audio profile, plus any MiniMax Music 3 row the audio
/// catalog does not carry, marking the ones installed.
pub fn list_audio(store: &Store) -> Result<(), Error> {
    let audio = AudioCatalog::embedded().map_err(Error::Failed)?;
    println!(
        "  {:<36} {:<16} {:<26} {:>10}  {:<10} {:<10} LICENSE TERMS",
        "ALIAS", "TASK", "FAMILY", "SIZE", "RUNS IN", "STATUS"
    );
    let mut seen = Vec::new();
    for profile in audio.entries() {
        let alias = &profile.identity.alias;
        seen.push(alias.clone());
        let installed = if store.audio_install_path(alias).is_dir() {
            "installed"
        } else {
            "available"
        };
        let terms = if profile.acknowledgments.is_empty() {
            String::new()
        } else {
            let ids: Vec<&str> = profile
                .acknowledgments
                .iter()
                .map(|a| a.id.as_str())
                .collect();
            format!("--accept {}", ids.join(" --accept "))
        };
        println!(
            "  {:<36} {:<16} {:<26} {:>10}  {:<10} {:<10} {}",
            alias,
            profile.identity.task.as_str(),
            profile.family,
            human_bytes(profile.download_bytes),
            runs_where(profile),
            installed,
            terms
        );
    }
    let music = catalog::embedded_music_entries().map_err(Error::Failed)?;
    for entry in music.iter().filter(|e| !seen.contains(&e.alias)) {
        let installed = if store.audio_install_path(&entry.alias).is_dir() {
            "installed"
        } else {
            "available"
        };
        println!(
            "  {:<36} {:<16} {:<26} {:>10}  {:<10} {:<10}",
            entry.alias, "music", "minimax_music3", "", "rust", installed
        );
    }
    Ok(())
}

/// Installs one pinned audio profile through the shared authenticated Hub
/// transport and its verified staging lifecycle, into `--library` or the
/// default library, settled before any request. `accepted` lists the
/// acknowledgment ids the caller passed with `--accept`.
pub fn pull_audio(
    catalog: &Catalog,
    store: &Store,
    client: &Client,
    alias: &str,
    options: &Options,
) -> Result<(), Error> {
    let audio = AudioCatalog::embedded().map_err(Error::Failed)?;
    let profile = audio.get(alias);
    let music_only = profile.is_none()
        && catalog::embedded_music_entries()
            .map_err(Error::Failed)?
            .iter()
            .any(|e| e.alias == alias);
    if profile.is_none() && !music_only {
        return Err(Error::Failed(format!(
            "no pinned audio profile named {alias:?}; `turbospark-model list-audio` lists them"
        )));
    }
    if let Some(profile) = profile {
        if profile.derived.is_some() {
            return Err(Error::Failed(format!(
                "{alias} converts its files on install; use `turbospark-model pull-drama3`"
            )));
        }
        // Terms first, then the check, so the refusal shows what was refused.
        for ack in &profile.acknowledgments {
            println!("[{}] {}\n", ack.id, ack.summary);
        }
        let missing: Vec<&str> = profile
            .acknowledgments
            .iter()
            .filter(|a| !options.accept.contains(&a.id))
            .map(|a| a.id.as_str())
            .collect();
        if !missing.is_empty() {
            let flags: Vec<String> = missing.iter().map(|id| format!("--accept {id}")).collect();
            return Err(Error::Failed(format!(
                "{alias} has terms to accept. Pass {} to accept them and install.",
                flags.join(" ")
            )));
        }
    }
    // An id the row does not carry is a typo or the wrong model: refuse it
    // rather than install while the caller believes terms were accepted.
    let carried: Vec<&str> = profile
        .map(|p| p.acknowledgments.iter().map(|a| a.id.as_str()).collect())
        .unwrap_or_default();
    if let Some(stray) = options
        .accept
        .iter()
        .find(|id| !carried.contains(&id.as_str()))
    {
        return Err(Error::Failed(format!(
            "{alias} has no terms with id {stray:?}"
        )));
    }
    let (target, env) =
        destination::library_target(store, InstallKind::Audio, alias, options.library.as_deref())?;

    INTERRUPTED.store(false, Ordering::Release);
    install_sigint_handler();
    let cancel = catalog::CancelFlag::new();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watcher_done = Arc::clone(&done);
    let watcher_cancel = cancel.clone();
    let watcher = thread::spawn(move || {
        while !watcher_done.load(Ordering::Acquire) {
            if INTERRUPTED.load(Ordering::Acquire) {
                watcher_cancel.cancel();
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
    });
    let mut last_path = None::<String>;
    let mut progress = |state: catalog::HubDownloadProgress| {
        if state.current_path != last_path {
            if let Some(path) = &state.current_path {
                eprintln!("[pull-audio] downloading {path}");
            }
            last_path = state.current_path;
        }
        if state.completed_files == state.total_files && state.total_files > 0 {
            eprintln!(
                "[pull-audio] verified {} file(s), {}",
                state.completed_files,
                human_bytes(state.completed_bytes)
            );
        }
    };
    // Music keeps its own promote step; every other row takes the shared one.
    let result = match profile.filter(|p| p.family != "minimax_music3") {
        Some(profile) => audio.install_acknowledged_into(
            client,
            catalog,
            store,
            &profile.identity,
            &options.accept,
            target.library_id.as_deref(),
            &env,
            &mut progress,
            &cancel,
        ),
        None => catalog::install_music_into(
            client,
            catalog,
            store,
            alias,
            target.library_id.as_deref(),
            &env,
            &mut progress,
            &cancel,
        ),
    };
    done.store(true, Ordering::Release);
    watcher
        .join()
        .map_err(|_| Error::Failed("audio pull cancellation watcher failed".to_string()))?;
    let installed = result.map_err(destination::failure)?;
    println!(
        "\ninstalled {} ({}) to {}",
        installed.alias,
        human_bytes(installed.install_bytes),
        installed.path.display()
    );
    if let Some(profile) = profile {
        if profile.capabilities.backend == "swift_coreml" {
            println!("note: this model runs in the TurboSpark app, not in the Rust server.");
        }
    }
    Ok(())
}

/// The audio install directory, or `None` when the alias is not an installed
/// audio row, so the caller can say so once for every kind.
pub fn path_audio(store: &Store, alias: &str) -> Option<std::path::PathBuf> {
    let path = store.audio_install_path(alias);
    path.is_dir().then_some(path)
}

/// Delete an audio install. Catalog rows go through `AudioCatalog::delete`,
/// which checks ownership before it removes anything; a music-only row has no
/// catalog identity, so it is removed as a managed directory.
pub fn remove_audio(store: &Store, alias: &str, yes: bool) -> Result<(), Error> {
    let path = store.audio_install_path(alias);
    if !path.is_dir() {
        return Err(Error::Failed(format!("{alias} is not installed")));
    }
    let bytes = catalog::directory_bytes(&path);
    if !yes {
        eprintln!(
            "delete {} ({})? This cannot be undone and re-installing means \
             downloading the model again.",
            path.display(),
            human_bytes(bytes)
        );
        eprint!("type the alias to confirm: ");
        use std::io::Write;
        let _ = std::io::stderr().flush();
        let mut typed = String::new();
        std::io::stdin()
            .read_line(&mut typed)
            .map_err(|e| Error::Failed(format!("reading confirmation: {e}")))?;
        if typed.trim() != alias {
            return Err(Error::Failed("not confirmed, nothing deleted".to_string()));
        }
    }
    let audio = AudioCatalog::embedded().map_err(Error::Failed)?;
    let pb = progress::spinner(format!("removing {}...", path.display()));
    // A music install carries no audio receipt, so `delete` would refuse it.
    let result = match audio.get(alias).filter(|p| p.family != "minimax_music3") {
        Some(profile) => audio.delete(store, &profile.identity),
        None => std::fs::remove_dir_all(&path)
            .map_err(|e| format!("removing {}: {e}", path.display()))
            .and_then(|()| store.forget_audio(alias)),
    };
    pb.finish_and_clear();
    result.map_err(Error::Failed)?;
    println!("removed {} ({})", path.display(), human_bytes(bytes));
    Ok(())
}
