//! `pull-drama3`: installs Fish Audio Drama 3 from its pinned MLX conversion.
//!
//! The install is a conversion, not a copy: the shards stream through
//! `repack::write_fish_drama3_install_streamed` by range read and are never
//! stored whole, so the 20 GB checkpoint is not on disk twice. Only the five
//! root files are downloaded as files, each with a pinned size and sha256.
//! The shards are not hash-checked (a range read never sees a whole file);
//! each shard's header must stay inside its pinned size, which ties the walk
//! to the pinned revision. Pins and license terms: `docs/FISH_DRAMA3_PHASE0.md`.
//!
//! The install is built in a sibling staging directory and renamed into place
//! once every file is present, so a failed or interrupted run never leaves a
//! half-built `--out`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use catalog::{Client, RepoRef};
use repack::{
    canonicalize_drama3_header, fetch_safetensors_header, write_fish_drama3_install_streamed,
    Gemma4Shards, HttpRangeSource, RangeSource, DRAMA3_SIDE_FILE, DRAMA3_TRUNK_DIR,
};

use super::human_bytes;
use crate::{Error, Options};

pub const REPO: &str = "mlx-community/fish-audio-drama-3-4bit";
pub const REVISION: &str = "cc639e27aba7edb73429d3e3806f471dcd79d316";

const GIB: u64 = 1 << 30;
/// Free space the walk needs: the install plus the trunk writer's own
/// staging copy. Measured on the real install (22 GiB result).
pub const WORKING_BYTES: u64 = 25 * GIB;
/// What the finished install occupies.
pub const RESULT_BYTES: u64 = 22 * GIB;

/// The Fish Audio Community License, as `docs/FISH_DRAMA3_PHASE0.md` records
/// it. The full text is installed as `LICENSE`.
pub const TERMS: &str = "\
Fish Audio Drama 3 is licensed under the Fish Audio Community License 1.0:
  - Free for research and non-commercial use.
  - Commercial use only while your annual revenue is under USD 500,000.
  - No hosted service (an API or app whose main purpose is text to speech
    for third parties) without a separate license from Fish Audio.
  - Every copy you distribute carries LICENSE and NOTICE.
  - A product that uses the model shows \"Built with Fish Audio\".
The full text is installed next to the model as LICENSE.";

#[derive(Debug, Clone)]
pub struct Pin {
    pub name: String,
    pub bytes: u64,
    /// Lowercase hex sha256, checked after the download.
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub name: String,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Pins {
    pub shards: Vec<Shard>,
    pub config: Pin,
    /// LICENSE, NOTICE and the tokenizer: small, fetched before the walk.
    pub small: Vec<Pin>,
    /// The codec, fetched last so a failed walk wastes no 1.5 GB download.
    pub codec: Pin,
}

fn pin(name: &str, bytes: u64, sha256: &str) -> Pin {
    Pin {
        name: name.to_string(),
        bytes,
        sha256: sha256.to_string(),
    }
}

impl Pins {
    /// `mlx-community/fish-audio-drama-3-4bit` at [`REVISION`]. Sizes and
    /// hashes are the table in `docs/FISH_DRAMA3_PHASE0.md`; `config.json`,
    /// `LICENSE` and `NOTICE` were hashed from the revision itself (the config
    /// matches its git blob fa8042126101484c6eb000a70a72e4be89eb839d).
    pub fn upstream() -> Self {
        let shard_bytes = [
            4_354_661_374,
            4_360_386_773,
            4_304_379_853,
            4_407_528_323,
            4_425_623_733,
            335_374_378,
        ];
        Self {
            shards: shard_bytes
                .iter()
                .enumerate()
                .map(|(i, &bytes)| Shard {
                    name: format!("model-{:05}.safetensors", i + 1),
                    bytes,
                })
                .collect(),
            config: pin(
                "config.json",
                53_444,
                "e980dfd9128edeb5f5d066fd7c322a2dfc08fedcbea58019d8d2e8e5cf7b50d5",
            ),
            small: vec![
                pin(
                    "LICENSE",
                    19_685,
                    "733a32442087acbda12ecafe8e71eb8ee5890c89b50ce7ac4ea3777faa3c823e",
                ),
                pin(
                    "NOTICE",
                    910,
                    "d2dc9c9fd71c2faf01b825f69797b20e1eca961505aa9ba2a3c5004b99262d73",
                ),
                pin(
                    "tokenizer.json",
                    20_791_070,
                    "0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6",
                ),
            ],
            codec: pin(
                "codec.safetensors",
                1_572_080_341,
                "6ff9d6f07920b2155e3a48ada21956004b96df9114d255ed9f0788484ba73040",
            ),
        }
    }
}

/// Where the bytes come from. The real one reads Hugging Face by range; tests
/// read memory.
pub trait Source {
    fn shard(&self, name: &str) -> Box<dyn RangeSource + '_>;
    /// Writes a whole root file to `dest`, which must not exist.
    fn root_file(&self, pin: &Pin, dest: &Path) -> Result<(), String>;
}

struct HubSource {
    repo: RepoRef,
    token: Option<String>,
}

impl Source for HubSource {
    fn shard(&self, name: &str) -> Box<dyn RangeSource + '_> {
        // No cache directory: the walk reads each shard once, and a cache
        // would store the 20 GB a second time.
        Box::new(
            HttpRangeSource::new(self.repo.file_url(name)).with_optional_token(self.token.clone()),
        )
    }

    fn root_file(&self, pin: &Pin, dest: &Path) -> Result<(), String> {
        HttpRangeSource::new(self.repo.file_url(&pin.name))
            .with_optional_token(self.token.clone())
            .download_to(dest, pin.bytes)
            .map(|_| ())
            .map_err(|e| format!("downloading {}: {e}", pin.name))
    }
}

/// `pull-drama3 --out DIR --accept-fish-license`.
pub fn pull_drama3(client: &Client, options: &Options) -> Result<(), Error> {
    println!("{TERMS}\n");
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
    refuse_existing(&dest)?;
    let available = available_bytes(&dest)?;
    let source = HubSource {
        repo: RepoRef::new(REPO, REVISION),
        token: client.token().map(str::to_string),
    };
    let started = Instant::now();
    install(&dest, &Pins::upstream(), &source, available, |stage| {
        eprintln!("[{:7.1}s] {stage}", started.elapsed().as_secs_f64());
    })?;
    println!(
        "\ninstalled Fish Audio Drama 3 ({}) to {}\nBuilt with Fish Audio.",
        human_bytes(RESULT_BYTES),
        dest.display()
    );
    Ok(())
}

/// The install, every outside effect passed in.
pub fn install(
    dest: &Path,
    pins: &Pins,
    source: &dyn Source,
    available: u64,
    mut progress: impl FnMut(&str),
) -> Result<(), Error> {
    refuse_existing(dest)?;
    check_free_space(available, WORKING_BYTES, dest)?;
    let staging = staging_dir(dest)?;
    if std::fs::symlink_metadata(&staging).is_ok() {
        return Err(Error::Failed(format!(
            "{} is left over from an interrupted install; remove it and run again",
            staging.display()
        )));
    }
    std::fs::create_dir_all(&staging)
        .map_err(|e| Error::Failed(format!("creating {}: {e}", staging.display())))?;
    let built = build(&staging, pins, source, &mut progress)
        .and_then(|()| verify_layout(&staging))
        .and_then(|()| {
            // Last look: rename onto an empty directory would succeed.
            refuse_existing(dest)?;
            std::fs::rename(&staging, dest)
                .map_err(|e| Error::Failed(format!("moving the install into place: {e}")))
        });
    if built.is_err() {
        // The staging directory is this run's own; nothing else writes it.
        let _ = std::fs::remove_dir_all(&staging);
    }
    built
}

fn build(
    staging: &Path,
    pins: &Pins,
    source: &dyn Source,
    progress: &mut impl FnMut(&str),
) -> Result<(), Error> {
    fetch_verified(source, &pins.config, staging, progress)?;
    for pin in &pins.small {
        fetch_verified(source, pin, staging, progress)?;
    }
    let config_text = std::fs::read_to_string(staging.join(&pins.config.name))
        .map_err(|e| Error::Failed(format!("reading config.json: {e}")))?;

    let shard_sources: Vec<_> = pins.shards.iter().map(|s| source.shard(&s.name)).collect();
    let mut headers = Vec::with_capacity(pins.shards.len());
    for (shard, reader) in pins.shards.iter().zip(&shard_sources) {
        let mut header = fetch_safetensors_header(reader.as_ref())
            .map_err(|e| Error::Failed(format!("reading the header of {}: {e}", shard.name)))?;
        check_shard_extent(&header, shard)?;
        canonicalize_drama3_header(&mut header)
            .map_err(|e| Error::Failed(format!("{}: {e}", shard.name)))?;
        headers.push(header);
    }
    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(&shard_sources)
            .map(|(h, s)| (h, s.as_ref()))
            .collect(),
    )
    .map_err(|e| Error::Failed(format!("assembling the shards: {e}")))?;
    let repo = RepoRef::new(REPO, REVISION);
    let model_id = repo.file_url("").trim_end_matches('/').to_string();
    let arch = write_fish_drama3_install_streamed(staging, &config_text, &model_id, &shards, |s| {
        progress(s)
    })
    .map_err(|e| Error::Failed(format!("converting the checkpoint: {e}")))?;
    progress(&format!(
        "converted {} layers, {} experts, hidden {}",
        arch.num_layers, arch.num_experts, arch.hidden_size
    ));
    fetch_verified(source, &pins.codec, staging, progress)
}

/// A shard whose header declares tensor data past the pinned file size is not
/// the pinned file.
fn check_shard_extent(header: &repack::SafetensorsHeader, shard: &Shard) -> Result<(), Error> {
    let end = header
        .tensors
        .values()
        .map(|t| t.data_offsets.1)
        .max()
        .unwrap_or(0);
    let declared = header.data_region_start().saturating_add(end);
    if declared > shard.bytes {
        return Err(Error::Failed(format!(
            "{} declares {declared} bytes but the pin is {}; it is not the pinned file",
            shard.name, shard.bytes
        )));
    }
    Ok(())
}

fn fetch_verified(
    source: &dyn Source,
    pin: &Pin,
    dir: &Path,
    progress: &mut impl FnMut(&str),
) -> Result<(), Error> {
    progress(&format!(
        "downloading {} ({})",
        pin.name,
        human_bytes(pin.bytes)
    ));
    let dest = dir.join(&pin.name);
    source.root_file(pin, &dest).map_err(Error::Failed)?;
    let actual = model_io::hash_file(&dest, 1 << 20)
        .map_err(|e| Error::Failed(format!("hashing {}: {e}", pin.name)))?;
    if actual != pin.sha256 {
        let _ = std::fs::remove_file(&dest);
        return Err(Error::Failed(format!(
            "{} has sha256 {actual}, expected {}",
            pin.name, pin.sha256
        )));
    }
    Ok(())
}

/// What `fish_drama3_say` and `turbospark-speak` open.
pub fn verify_layout(dir: &Path) -> Result<(), Error> {
    let trunk = Path::new(DRAMA3_TRUNK_DIR);
    for relative in [
        PathBuf::from("config.json"),
        PathBuf::from("tokenizer.json"),
        PathBuf::from("codec.safetensors"),
        PathBuf::from("LICENSE"),
        PathBuf::from("NOTICE"),
        PathBuf::from(DRAMA3_SIDE_FILE),
        trunk.join("manifest.json"),
        trunk.join("trunk_role.json"),
    ] {
        if !dir.join(&relative).is_file() {
            return Err(Error::Failed(format!(
                "the install is missing {}",
                relative.display()
            )));
        }
    }
    Ok(())
}

pub fn refuse_existing(dest: &Path) -> Result<(), Error> {
    if std::fs::symlink_metadata(dest).is_ok() {
        return Err(Error::Failed(format!(
            "{} already exists; this command never overwrites. Remove it or pick another --out",
            dest.display()
        )));
    }
    Ok(())
}

pub fn check_free_space(available: u64, required: u64, dest: &Path) -> Result<(), Error> {
    if available < required {
        return Err(Error::Failed(format!(
            "installing Drama 3 needs about {} free ({} for the install, the rest working space); \
             {} is available where {} will be created",
            human_bytes(required),
            human_bytes(RESULT_BYTES),
            human_bytes(available),
            dest.display()
        )));
    }
    Ok(())
}

/// A sibling of `dest`, so the final rename stays on one filesystem.
fn staging_dir(dest: &Path) -> Result<PathBuf, Error> {
    let name = dest
        .file_name()
        .ok_or_else(|| Error::Failed(format!("{} has no directory name", dest.display())))?;
    let parent = dest.parent().filter(|p| !p.as_os_str().is_empty());
    let mut staging_name = std::ffi::OsString::from(".");
    staging_name.push(name);
    staging_name.push(".partial");
    Ok(parent.unwrap_or_else(|| Path::new(".")).join(staging_name))
}

/// Bytes a non-root user can still write where `path` will be created: the
/// nearest existing ancestor's filesystem.
#[cfg(unix)]
pub fn available_bytes(path: &Path) -> Result<u64, Error> {
    use std::os::unix::ffi::OsStrExt;
    let mut probe = path.to_path_buf();
    if probe.is_relative() {
        probe = std::env::current_dir()
            .map_err(|e| Error::Failed(format!("reading the working directory: {e}")))?
            .join(probe);
    }
    while !probe.exists() {
        if !probe.pop() {
            return Err(Error::Failed(format!(
                "no existing parent directory for {}",
                path.display()
            )));
        }
    }
    let c_path = std::ffi::CString::new(probe.as_os_str().as_bytes())
        .map_err(|_| Error::Failed("the path contains a NUL byte".to_string()))?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c_path` is a valid NUL-terminated string and `stats` is a
    // writable statvfs; the struct is read only after a zero return.
    let status = unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) };
    if status != 0 {
        return Err(Error::Failed(format!(
            "reading free space on {}: {}",
            probe.display(),
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: statvfs returned zero, so it filled the struct.
    let stats = unsafe { stats.assume_init() };
    // The field widths differ by platform (u32 and u64 on macOS), so the
    // casts are not useless everywhere.
    #[allow(clippy::unnecessary_cast)]
    let free = (stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64);
    Ok(free)
}

#[cfg(not(unix))]
pub fn available_bytes(_path: &Path) -> Result<u64, Error> {
    Err(Error::Failed(
        "the free-disk check needs a unix platform".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use repack::{
        synthetic_fish_drama3_checkpoint, Drama3NormStorage, MemoryRangeSource, SyntheticDrama3,
    };

    impl std::fmt::Debug for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Error::Usage(m) => write!(f, "Usage({m})"),
                Error::Failed(m) => write!(f, "Failed({m})"),
            }
        }
    }

    fn message(error: Error) -> String {
        match error {
            Error::Usage(m) | Error::Failed(m) => m,
        }
    }

    fn tempdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pull_drama3_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The synthetic checkpoint behind a [`Source`], with tiny root files.
    struct Memory {
        checkpoint: SyntheticDrama3,
        files: Vec<(String, Vec<u8>)>,
        root_calls: AtomicUsize,
    }

    impl Source for Memory {
        fn shard(&self, _name: &str) -> Box<dyn RangeSource + '_> {
            Box::new(MemoryRangeSource::new(&self.checkpoint.blob))
        }

        fn root_file(&self, pin: &Pin, dest: &Path) -> Result<(), String> {
            self.root_calls.fetch_add(1, Ordering::SeqCst);
            let bytes = if pin.name == "config.json" {
                self.checkpoint.config_text.as_bytes().to_vec()
            } else {
                self.files
                    .iter()
                    .find(|(name, _)| *name == pin.name)
                    .map(|(_, bytes)| bytes.clone())
                    .ok_or_else(|| format!("no fixture for {}", pin.name))?
            };
            std::fs::write(dest, bytes).map_err(|e| e.to_string())
        }
    }

    fn fixture() -> (Memory, Pins) {
        let checkpoint = synthetic_fish_drama3_checkpoint(256, 4, Drama3NormStorage::Shifted);
        let files: Vec<(String, Vec<u8>)> = [
            ("LICENSE", b"license text".to_vec()),
            ("NOTICE", b"notice text".to_vec()),
            ("tokenizer.json", b"{}".to_vec()),
            ("codec.safetensors", b"codec bytes".to_vec()),
        ]
        .into_iter()
        .map(|(n, b)| (n.to_string(), b))
        .collect();
        let pin_of = |name: &str, bytes: &[u8]| Pin {
            name: name.to_string(),
            bytes: bytes.len() as u64,
            sha256: model_io::hash_data(bytes),
        };
        let find = |name: &str| files.iter().find(|(n, _)| n == name).unwrap().1.clone();
        let pins = Pins {
            shards: vec![Shard {
                name: "model-00001.safetensors".to_string(),
                bytes: checkpoint.blob.len() as u64,
            }],
            config: pin_of("config.json", checkpoint.config_text.as_bytes()),
            small: ["LICENSE", "NOTICE", "tokenizer.json"]
                .iter()
                .map(|n| pin_of(n, &find(n)))
                .collect(),
            codec: pin_of("codec.safetensors", &find("codec.safetensors")),
        };
        (
            Memory {
                checkpoint,
                files,
                root_calls: AtomicUsize::new(0),
            },
            pins,
        )
    }

    const PLENTY: u64 = u64::MAX;

    #[test]
    fn installs_a_layout_the_runtime_opens() {
        let root = tempdir("ok");
        let dest = root.join("drama3");
        let (source, pins) = fixture();
        let mut stages = Vec::new();
        install(&dest, &pins, &source, PLENTY, |s| {
            stages.push(s.to_string())
        })
        .unwrap();
        verify_layout(&dest).unwrap();
        assert_eq!(
            runtime::native_audio::tts_model_type(&dest).as_deref(),
            Some(runtime::native_audio::FISH_DRAMA3_FAMILY)
        );
        // LICENSE and NOTICE travel with the install, byte for byte.
        assert_eq!(
            std::fs::read(dest.join("LICENSE")).unwrap(),
            b"license text"
        );
        assert_eq!(std::fs::read(dest.join("NOTICE")).unwrap(), b"notice text");
        // The staging directory is gone and the walk reported its stages.
        assert!(!staging_dir(&dest).unwrap().exists());
        assert!(stages.iter().any(|s| s.contains("codec.safetensors")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn refuses_an_existing_destination_before_any_download() {
        let root = tempdir("exists");
        let dest = root.join("drama3");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("keep.txt"), b"mine").unwrap();
        let (source, pins) = fixture();
        let err = install(&dest, &pins, &source, PLENTY, |_| {}).unwrap_err();
        assert!(message(err).contains("already exists"));
        assert_eq!(source.root_calls.load(Ordering::SeqCst), 0);
        assert_eq!(std::fs::read(dest.join("keep.txt")).unwrap(), b"mine");
        assert!(!staging_dir(&dest).unwrap().exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn refuses_too_little_disk_before_any_download() {
        let root = tempdir("disk");
        let dest = root.join("drama3");
        let (source, pins) = fixture();
        let err = install(&dest, &pins, &source, WORKING_BYTES - 1, |_| {}).unwrap_err();
        assert!(message(err).contains("free"));
        assert_eq!(source.root_calls.load(Ordering::SeqCst), 0);
        assert!(!dest.exists() && !staging_dir(&dest).unwrap().exists());
        // Exactly the requirement is enough.
        assert!(check_free_space(WORKING_BYTES, WORKING_BYTES, &dest).is_ok());
        assert!(check_free_space(WORKING_BYTES - 1, WORKING_BYTES, &dest).is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_wrong_hash_stops_the_install_and_leaves_no_directory() {
        let root = tempdir("hash");
        let dest = root.join("drama3");
        let (source, mut pins) = fixture();
        pins.small[0].sha256 = model_io::hash_data(b"a different license");
        let err = install(&dest, &pins, &source, PLENTY, |_| {}).unwrap_err();
        assert!(message(err).contains("LICENSE has sha256"));
        assert!(!dest.exists() && !staging_dir(&dest).unwrap().exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_wrong_codec_hash_fails_after_the_walk_and_cleans_up() {
        let root = tempdir("codec");
        let dest = root.join("drama3");
        let (source, mut pins) = fixture();
        pins.codec.sha256 = model_io::hash_data(b"not the codec");
        let err = install(&dest, &pins, &source, PLENTY, |_| {}).unwrap_err();
        assert!(message(err).contains("codec.safetensors has sha256"));
        assert!(!dest.exists() && !staging_dir(&dest).unwrap().exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_shard_larger_than_its_pin_is_refused() {
        let root = tempdir("extent");
        let dest = root.join("drama3");
        let (source, mut pins) = fixture();
        pins.shards[0].bytes -= 1;
        let err = install(&dest, &pins, &source, PLENTY, |_| {}).unwrap_err();
        assert!(message(err).contains("not the pinned file"));
        assert!(!dest.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_leftover_staging_directory_is_reported_and_kept() {
        let root = tempdir("stale");
        let dest = root.join("drama3");
        let stale = staging_dir(&dest).unwrap();
        std::fs::create_dir_all(&stale).unwrap();
        let (source, pins) = fixture();
        let err = install(&dest, &pins, &source, PLENTY, |_| {}).unwrap_err();
        assert!(message(err).contains("interrupted install"));
        assert!(
            stale.exists(),
            "another run's staging directory is not ours to delete"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn the_layout_check_names_a_missing_file() {
        let root = tempdir("layout");
        assert!(message(verify_layout(&root).unwrap_err()).contains("config.json"));
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
        for term in [
            "USD 500,000",
            "hosted service",
            "LICENSE and NOTICE",
            "Built with Fish Audio",
            "non-commercial",
        ] {
            assert!(TERMS.contains(term), "the terms omit {term:?}");
        }
        assert!(TERMS.is_ascii());
    }

    #[test]
    fn staging_is_a_sibling_of_the_destination() {
        assert_eq!(
            staging_dir(Path::new("/a/b/drama3")).unwrap(),
            Path::new("/a/b/.drama3.partial")
        );
        assert_eq!(
            staging_dir(Path::new("drama3")).unwrap(),
            Path::new("./.drama3.partial")
        );
    }

    #[test]
    fn upstream_pins_match_the_recorded_table() {
        let pins = Pins::upstream();
        assert_eq!(pins.shards.len(), 6);
        assert_eq!(pins.shards[0].name, "model-00001.safetensors");
        let total: u64 = pins.shards.iter().map(|s| s.bytes).sum();
        assert_eq!(total, 22_187_954_434);
        assert_eq!(REVISION.len(), 40);
        for pin in pins.small.iter().chain([&pins.config, &pins.codec]) {
            assert_eq!(pin.sha256.len(), 64, "{}", pin.name);
        }
    }

    #[cfg(unix)]
    #[test]
    fn free_space_reads_the_nearest_existing_ancestor() {
        let root = tempdir("avail");
        let deep = root.join("not").join("yet").join("made");
        let at_root = available_bytes(&root).unwrap();
        let at_deep = available_bytes(&deep).unwrap();
        assert!(at_root > 0);
        // Same filesystem, so within a whisker of each other.
        assert!(at_deep.abs_diff(at_root) < 1 << 30);
        let _ = std::fs::remove_dir_all(root);
    }
}
