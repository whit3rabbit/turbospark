//! `pull-drama3`: installs Fish Audio Drama 3 into a directory of your choice.
//!
//! The walk itself, the pins and the license text live in the catalog
//! (`catalog::drama3_install` and the `fish-drama3-4bit` audio profile), so
//! this command, the FFI and the app install the same thing. This wrapper adds
//! the terms prompt, the destination rules and Ctrl-C handling. An install
//! made here is a plain directory for `fish_drama3_say` and `turbospark-speak
//! --model`; the app's managed store keeps its own receipt-backed install.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use catalog::drama3_install as engine;
use catalog::{AudioCatalog, CancelFlag, Client};

use super::{human_bytes, install_audio_sigint_handler, AUDIO_PULL_INTERRUPTED};
use crate::{Error, Options};

/// The id of the license acknowledgment on the catalog profile.
const LICENSE_ACK: &str = "fish_audio_community_license";

/// `pull-drama3 --out DIR --accept-fish-license`.
pub fn pull_drama3(client: &Client, options: &Options) -> Result<(), Error> {
    let catalog = AudioCatalog::embedded().map_err(Error::Failed)?;
    let profile = catalog
        .get(engine::ALIAS)
        .ok_or_else(|| Error::Failed("the audio catalog has no Drama 3 profile".to_string()))?;
    let terms = profile
        .acknowledgments
        .iter()
        .find(|a| a.id == LICENSE_ACK)
        .map(|a| a.summary.as_str())
        .ok_or_else(|| Error::Failed("the Drama 3 profile lost its license terms".to_string()))?;
    println!("{terms}\n");
    if !options.accept_fish_license {
        return Err(Error::Usage(
            "pass --accept-fish-license to accept these terms and install".to_string(),
        ));
    }
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| Error::Usage("pull-drama3 needs --out <DIR>".to_string()))?;
    let dest = PathBuf::from(out);
    // Refuse before the disk probe or the network: the cheap answer first.
    engine::refuse_existing(&dest).map_err(Error::Failed)?;
    let available = model_io::available_bytes(&dest).map_err(Error::Failed)?;
    let pins = engine::Pins::from_assets(&profile.assets).map_err(Error::Failed)?;

    AUDIO_PULL_INTERRUPTED.store(false, Ordering::Release);
    install_audio_sigint_handler();
    let cancel = CancelFlag::new();
    let done = Arc::new(AtomicBool::new(false));
    let watcher = {
        let (done, cancel) = (Arc::clone(&done), cancel.clone());
        thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                if AUDIO_PULL_INTERRUPTED.load(Ordering::Acquire) {
                    cancel.cancel();
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
        })
    };
    let source = engine::HubSource::new(client.token(), &cancel);
    let started = Instant::now();
    let result = engine::install(&dest, &pins, &source, available, &cancel, &mut |p| {
        eprintln!("[{:7.1}s] {}", started.elapsed().as_secs_f64(), p.stage);
    });
    done.store(true, Ordering::Release);
    let _ = watcher.join();
    result.map_err(|e| {
        if AUDIO_PULL_INTERRUPTED.load(Ordering::Acquire) {
            Error::Failed("interrupted; nothing was left behind".to_string())
        } else {
            Error::Failed(e)
        }
    })?;
    println!(
        "\ninstalled Fish Audio Drama 3 ({}) to {}\nBuilt with Fish Audio.",
        human_bytes(engine::RESULT_BYTES),
        dest.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use repack::{
        synthetic_fish_drama3_checkpoint, Drama3NormStorage, MemoryRangeSource, RangeSource,
        SyntheticDrama3,
    };

    /// The synthetic checkpoint behind the catalog walk's [`engine::Source`].
    struct Memory {
        checkpoint: SyntheticDrama3,
    }

    impl engine::Source for Memory {
        fn shard(&self, _name: &str) -> Box<dyn RangeSource + '_> {
            Box::new(MemoryRangeSource::new(&self.checkpoint.blob))
        }

        fn root_file(&self, pin: &engine::Pin, dest: &Path) -> Result<(), String> {
            let bytes: Vec<u8> = match pin.name.as_str() {
                "config.json" => self.checkpoint.config_text.as_bytes().to_vec(),
                "LICENSE" => b"license text".to_vec(),
                "NOTICE" => b"notice text".to_vec(),
                "tokenizer.json" => b"{}".to_vec(),
                "codec.safetensors" => b"codec bytes".to_vec(),
                other => return Err(format!("no fixture for {other}")),
            };
            std::fs::write(dest, bytes).map_err(|e| e.to_string())
        }
    }

    fn pin_of(name: &str, bytes: &[u8]) -> engine::Pin {
        engine::Pin {
            name: name.to_string(),
            bytes: bytes.len() as u64,
            sha256: model_io::hash_data(bytes),
        }
    }

    /// What the runtime opens: the converter's layout and a root config the
    /// runtime recognises as Drama 3.
    #[test]
    fn installs_a_layout_the_runtime_opens() {
        let root = std::env::temp_dir().join(format!("pull_drama3_ok_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let dest = root.join("drama3");
        let checkpoint = synthetic_fish_drama3_checkpoint(256, 4, Drama3NormStorage::Shifted);
        let pins = engine::Pins {
            shards: vec![engine::Shard {
                name: "model-00001.safetensors".to_string(),
                bytes: checkpoint.blob.len() as u64,
            }],
            config: pin_of("config.json", checkpoint.config_text.as_bytes()),
            small: vec![
                pin_of("LICENSE", b"license text"),
                pin_of("NOTICE", b"notice text"),
                pin_of("tokenizer.json", b"{}"),
            ],
            codec: pin_of("codec.safetensors", b"codec bytes"),
        };
        let source = Memory { checkpoint };
        engine::install(
            &dest,
            &pins,
            &source,
            u64::MAX,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        engine::verify_layout(&dest).unwrap();
        assert_eq!(
            runtime::native_audio::tts_model_type(&dest).as_deref(),
            Some(runtime::native_audio::FISH_DRAMA3_FAMILY)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn the_license_flag_gates_everything_and_the_terms_are_complete() {
        // Refused before any network or disk work: no --out, no staging.
        let without = Options::default();
        let Err(Error::Usage(m)) = pull_drama3(&Client::new(), &without) else {
            panic!("an install started without the license flag");
        };
        assert!(m.contains("--accept-fish-license"), "{m}");
        // Accepting still needs a destination.
        let accepted = Options {
            accept_fish_license: true,
            ..Options::default()
        };
        let Err(Error::Usage(m)) = pull_drama3(&Client::new(), &accepted) else {
            panic!("an install started without --out");
        };
        assert!(m.contains("--out"), "{m}");
        // The terms it prints are the catalog's, and carry every obligation.
        let catalog = AudioCatalog::embedded().unwrap();
        let terms = &catalog.get(engine::ALIAS).unwrap().acknowledgments[0].summary;
        for term in [
            "USD 500,000",
            "hosted service",
            "LICENSE and NOTICE",
            "Built with Fish Audio",
            "non-commercial",
        ] {
            assert!(terms.contains(term), "the terms omit {term:?}");
        }
    }

    #[test]
    fn an_existing_destination_is_refused_before_the_network() {
        let root = std::env::temp_dir().join(format!("pull_drama3_exists_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let options = Options {
            accept_fish_license: true,
            out: Some(root.display().to_string()),
            ..Options::default()
        };
        let Err(Error::Failed(m)) = pull_drama3(&Client::new(), &options) else {
            panic!("an existing destination was accepted");
        };
        assert!(m.contains("already exists"), "{m}");
        let _ = std::fs::remove_dir_all(root);
    }
}
