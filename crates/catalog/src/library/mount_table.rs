//! The inputs and outcomes of deciding whether a path may host a library.
//!
//! Every OS-derived fact arrives through [`LocationEnv`], so tests decide
//! locations against a fixed mount table and a temp volumes folder without
//! spawning a process.

use super::locate::volume_root_of;
use std::ffi::OsString;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// One line of the mount table: `<device> on <mount point> (<fstype>, ...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountEntry {
    pub mount_point: PathBuf,
    pub fs_type: String,
    /// The `local` flag. Its absence marks a network mount.
    pub local: bool,
    pub read_only: bool,
}

/// Why a location cannot host a library. Each kind maps to its own sentence
/// in the CLI and the app, so a refusal names the limit it hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocationRefusal {
    /// This Mac already covers the home folder's disk (1.10).
    SameDiskAsHome,
    /// Libraries must be on a directly connected drive (1.4).
    Network,
    /// The filesystem cannot hold a single file of 4 GiB or more (1.5).
    SmallFileLimit {
        fs_type: String,
    },
    ReadOnly,
    NotADirectory,
    /// Inside the named library (1.7).
    NestedIn {
        library: String,
    },
    /// Inside the TurboSpark home folder (1.7).
    InsideHome,
    /// No mount table entry has the location's volume root as its mount
    /// point, the table could not be read, or the location is a mount nested
    /// inside another volume. The drive's filesystem and connection are then
    /// unknown, so the location cannot be shown to pass the network and
    /// file-size rules.
    UnknownVolume,
}

/// Where mount entries come from. `System` runs `/sbin/mount` only when
/// `validate_location` or a relocating `locate` asks, never at construction,
/// because the default store is built on server request paths and must not
/// spawn or fail there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MountSource {
    System,
    Fixed(Vec<MountEntry>),
}

/// Every OS-derived input to location decisions, injectable for tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocationEnv {
    pub mount_source: MountSource,
    /// `/Volumes` on a live system, a temp folder of fake volumes in tests.
    pub volumes_root: PathBuf,
    /// `st_dev` of the home folder, for the same-disk refusal.
    pub home_device: u64,
}

impl LocationEnv {
    /// The live environment for the home at `home`: `/Volumes` and the home
    /// folder's device. Cheap, and spawns no process, because the default
    /// store is built on server request paths.
    ///
    /// Debug builds only: `TURBOSPARK_TEST_MOUNT_TABLE` (a file of mount
    /// table text), `TURBOSPARK_TEST_VOLUMES_ROOT`, and
    /// `TURBOSPARK_TEST_HOME_DEVICE` override the three inputs. Release
    /// builds ignore them, so a shipped binary cannot bypass the network
    /// refusal.
    pub fn system(home: &Path) -> Result<Self, String> {
        Self::system_with(home, |name| std::env::var_os(name))
    }

    fn system_with(home: &Path, var: impl Fn(&str) -> Option<OsString>) -> Result<Self, String> {
        let overrides = test_overrides(var)?;
        let mount_source = match overrides.mount_table {
            Some(file) => {
                let text = std::fs::read_to_string(&file)
                    .map_err(|e| format!("reading {}: {e}", file.display()))?;
                MountSource::Fixed(parse_mount_table(&text))
            }
            None => MountSource::System,
        };
        let home_device = match overrides.home_device {
            Some(device) => device,
            None => device_of_nearest_existing(home)?,
        };
        Ok(Self {
            mount_source,
            volumes_root: overrides
                .volumes_root
                .unwrap_or_else(|| PathBuf::from("/Volumes")),
            home_device,
        })
    }

    /// Reads the mount table on demand: `validate_location`, and `locate`
    /// when the recorded root does not match.
    pub fn mount_table(&self) -> Result<Vec<MountEntry>, String> {
        match &self.mount_source {
            MountSource::Fixed(entries) => Ok(entries.clone()),
            MountSource::System => read_system_mount_table(),
        }
    }
}

#[derive(Debug, Default)]
struct TestOverrides {
    mount_table: Option<PathBuf>,
    volumes_root: Option<PathBuf>,
    home_device: Option<u64>,
}

/// The debug-only inputs that let integration tests stand in fake volumes.
/// An empty value counts as unset.
#[cfg(debug_assertions)]
fn test_overrides(var: impl Fn(&str) -> Option<OsString>) -> Result<TestOverrides, String> {
    let read = |name: &str| var(name).filter(|value| !value.is_empty());
    let home_device = match read("TURBOSPARK_TEST_HOME_DEVICE") {
        Some(raw) => Some(
            raw.to_str()
                .and_then(|text| text.trim().parse::<u64>().ok())
                .ok_or_else(|| {
                    format!("TURBOSPARK_TEST_HOME_DEVICE is not a device number: {raw:?}")
                })?,
        ),
        None => None,
    };
    Ok(TestOverrides {
        mount_table: read("TURBOSPARK_TEST_MOUNT_TABLE").map(PathBuf::from),
        volumes_root: read("TURBOSPARK_TEST_VOLUMES_ROOT").map(PathBuf::from),
        home_device,
    })
}

#[cfg(not(debug_assertions))]
fn test_overrides(_var: impl Fn(&str) -> Option<OsString>) -> Result<TestOverrides, String> {
    Ok(TestOverrides::default())
}

/// The device of `path`, or of its nearest existing ancestor. A first run
/// builds the default store before the home folder exists, and the home
/// will be created on that ancestor's device.
fn device_of_nearest_existing(path: &Path) -> Result<u64, String> {
    for candidate in path.ancestors() {
        match std::fs::metadata(candidate) {
            Ok(meta) => return Ok(meta.dev()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("reading {}: {e}", candidate.display())),
        }
    }
    Err(format!("no existing folder holds {}", path.display()))
}

fn read_system_mount_table() -> Result<Vec<MountEntry>, String> {
    #[cfg(test)]
    SYSTEM_MOUNT_READS.with(|reads| reads.set(reads.get() + 1));
    // An absolute path, so a PATH entry cannot substitute its own table.
    let output = Command::new("/sbin/mount")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("running /sbin/mount: {e}"))?;
    if !output.status.success() {
        return Err(format!("/sbin/mount exited with {}", output.status));
    }
    Ok(parse_mount_table(&String::from_utf8_lossy(&output.stdout)))
}

/// Parses `/sbin/mount` output, one `<device> on <mount point> (<fstype>,
/// <flags...>)` entry per line. Lines that do not have that shape are
/// skipped, so one odd mount cannot block every location decision.
pub fn parse_mount_table(text: &str) -> Vec<MountEntry> {
    text.lines().filter_map(parse_mount_line).collect()
}

fn parse_mount_line(line: &str) -> Option<MountEntry> {
    // The first ` on ` ends the device and the last ` (` starts the flags,
    // so a mount point may itself hold spaces, ` on `, or ` (`.
    let (_device, rest) = line.trim_end().split_once(" on ")?;
    let (mount_point, flags) = rest.rsplit_once(" (")?;
    if !mount_point.starts_with('/') {
        return None;
    }
    let mut flags = flags.strip_suffix(')')?.split(',').map(str::trim);
    let fs_type = flags.next().filter(|fs_type| !fs_type.is_empty())?;
    let mut entry = MountEntry {
        mount_point: PathBuf::from(mount_point),
        fs_type: fs_type.to_string(),
        local: false,
        read_only: false,
    };
    for flag in flags {
        match flag {
            "local" => entry.local = true,
            "read-only" => entry.read_only = true,
            _ => {}
        }
    }
    Some(entry)
}

/// Decides whether `path` may host a library, and returns its volume root.
///
/// `path` must exist and be canonical. The device check runs before the
/// mount table is read: every internal folder resolves to the volume root
/// `/`, whose entry is the sealed read-only system volume, so it must be
/// refused as the home's disk rather than as read-only. Nesting inside
/// another library or the home is the registry's check, because it needs
/// the library list.
pub fn validate_location(path: &Path, env: &LocationEnv) -> Result<PathBuf, LocationRefusal> {
    let meta = std::fs::metadata(path).map_err(|_| LocationRefusal::NotADirectory)?;
    if !meta.is_dir() {
        return Err(LocationRefusal::NotADirectory);
    }
    if meta.dev() == env.home_device {
        return Err(LocationRefusal::SameDiskAsHome);
    }
    let root = volume_root_of(path, env).map_err(|_| LocationRefusal::UnknownVolume)?;
    // The direct-child rule names a root by path alone. A mount nested in
    // that volume, such as a share at `/Volumes/T7/share`, is on another
    // device, and judging it by the outer volume's entry would let it past
    // the network and file-size rules.
    let root_device = std::fs::metadata(&root)
        .map_err(|_| LocationRefusal::UnknownVolume)?
        .dev();
    if root_device != meta.dev() {
        return Err(LocationRefusal::UnknownVolume);
    }
    let table = env
        .mount_table()
        .map_err(|_| LocationRefusal::UnknownVolume)?;
    let entry = mount_entry_at(&table, &root).ok_or(LocationRefusal::UnknownVolume)?;
    if !entry.local {
        return Err(LocationRefusal::Network);
    }
    // FAT32 caps a file at 4 GiB, below most model weights.
    if entry.fs_type == "msdos" {
        return Err(LocationRefusal::SmallFileLimit {
            fs_type: entry.fs_type.clone(),
        });
    }
    if entry.read_only {
        return Err(LocationRefusal::ReadOnly);
    }
    Ok(root)
}

/// The entry mounted at exactly `mount_point`. Equality, never a prefix:
/// `/Volumes/T7` must not decide for `/Volumes/T7 Backup`. The last entry
/// wins because a later mount on the same point hides the earlier one.
pub(super) fn mount_entry_at<'a>(
    table: &'a [MountEntry],
    mount_point: &Path,
) -> Option<&'a MountEntry> {
    table
        .iter()
        .rev()
        .find(|entry| entry.mount_point == mount_point)
}

#[cfg(test)]
thread_local! {
    /// Reads of the system mount table on this thread. Thread-local so a
    /// parallel test that reads the real table cannot disturb a no-spawn
    /// assertion.
    static SYSTEM_MOUNT_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Reads of the system mount table on this thread so far. Shared with the
/// registry and store tests, which count the reads of a whole refresh or
/// listing.
#[cfg(test)]
pub(crate) fn system_mount_reads() -> usize {
    SYSTEM_MOUNT_READS.with(|reads| reads.get())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Store;
    use std::collections::HashMap;
    use std::os::unix::fs::MetadataExt;

    fn entry(mount_point: &str, fs_type: &str, local: bool, read_only: bool) -> MountEntry {
        MountEntry {
            mount_point: PathBuf::from(mount_point),
            fs_type: fs_type.to_string(),
            local,
            read_only,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ts-library-mount-table-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // The temp dir is behind the /var symlink, and validation takes
        // canonical paths.
        dir.canonicalize().unwrap()
    }

    /// Fake volumes under `<base>/Volumes`, each with a `Models` folder, and
    /// a fixed mount table built from real-format lines so validation also
    /// runs through the parser. `home_device` differs from the temp dir's
    /// device, so the same-disk rule does not fire for fake volumes.
    struct FakeVolumes {
        base: PathBuf,
        volumes_root: PathBuf,
        env: LocationEnv,
    }

    impl FakeVolumes {
        fn new(test: &str, volumes: &[(&str, &str)]) -> Self {
            let base = scratch(test);
            let volumes_root = base.join("Volumes");
            let mut table = String::new();
            for (index, (name, flags)) in volumes.iter().enumerate() {
                let volume = volumes_root.join(name);
                std::fs::create_dir_all(volume.join("Models")).unwrap();
                table.push_str(&format!(
                    "/dev/disk{}s1 on {} ({flags})\n",
                    index + 20,
                    volume.display()
                ));
            }
            let entries = parse_mount_table(&table);
            assert_eq!(entries.len(), volumes.len(), "fixture table:\n{table}");
            let home_device = std::fs::metadata(&base).unwrap().dev().wrapping_add(1);
            Self {
                env: LocationEnv {
                    mount_source: MountSource::Fixed(entries),
                    volumes_root: volumes_root.clone(),
                    home_device,
                },
                base,
                volumes_root,
            }
        }

        fn volume(&self, name: &str) -> PathBuf {
            self.volumes_root.join(name)
        }

        fn validate(&self, volume: &str) -> Result<PathBuf, LocationRefusal> {
            validate_location(&self.volume(volume).join("Models"), &self.env)
        }
    }

    impl Drop for FakeVolumes {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.base);
        }
    }

    #[test]
    fn parses_apfs_hfs_exfat_msdos_smbfs_read_only_and_spaced_mount_points() {
        let text = "\
/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)
/dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled, nobrowse, protect, root data)
/dev/disk18s1 on /Volumes/T7 (hfs, local, nodev, nosuid, journaled, noowners)
/dev/disk19s1 on /Volumes/Embedding Gemma Transcribe (exfat, local, nodev, nosuid, noowners, noatime, fskit)
/dev/disk20s1 on /Volumes/NO NAME (msdos, local, nodev, nosuid, noowners, noatime, fskit)
//guest@nas.local/media on /Volumes/media (smbfs, nodev, nosuid, mounted by someone)
/dev/disk21s1 on /Volumes/Installer (hfs, local, nodev, nosuid, read-only, noowners, quarantine, mounted by someone)
";
        assert_eq!(
            parse_mount_table(text),
            vec![
                entry("/", "apfs", true, true),
                entry("/System/Volumes/Data", "apfs", true, false),
                entry("/Volumes/T7", "hfs", true, false),
                entry("/Volumes/Embedding Gemma Transcribe", "exfat", true, false),
                entry("/Volumes/NO NAME", "msdos", true, false),
                // No `local` flag: a network mount.
                entry("/Volumes/media", "smbfs", false, false),
                entry("/Volumes/Installer", "hfs", true, true),
            ]
        );
    }

    #[test]
    fn mount_point_runs_from_the_first_on_to_the_last_parenthesis() {
        // The device may hold spaces, and the mount point may hold both ` on `
        // and ` (`, so the split uses the first ` on ` and the last ` (`.
        let text = "map auto_home on /System/Volumes/Data/home (autofs, automounted, nobrowse)\n\
                    /dev/disk4s1 on /Volumes/Backup on Tuesday (old) (apfs, local, journaled)\n";
        assert_eq!(
            parse_mount_table(text),
            vec![
                entry("/System/Volumes/Data/home", "autofs", false, false),
                entry("/Volumes/Backup on Tuesday (old)", "apfs", true, false),
            ]
        );
    }

    #[test]
    fn parser_skips_malformed_lines() {
        let text = [
            "",
            "not a mount line",
            "/dev/disk5s1 on /Volumes/NoFlags",
            "/dev/disk6s1 on /Volumes/Open (hfs, local",
            "/dev/disk7s1 on /Volumes/NoType (, local)",
            "/dev/disk8s1 on  (hfs, local)",
            "/dev/disk9s1 on relative/path (hfs, local)",
            "/dev/disk18s1 on /Volumes/T7 (hfs, local, journaled)",
        ]
        .join("\n");
        assert_eq!(
            parse_mount_table(&text),
            vec![entry("/Volumes/T7", "hfs", true, false)]
        );
    }

    #[test]
    fn local_writable_volumes_are_accepted_and_return_their_volume_root() {
        let fake = FakeVolumes::new(
            "accept",
            &[
                ("T7", "hfs, local, nodev, nosuid, journaled, noowners"),
                ("Studio", "apfs, local, nodev, nosuid, journaled, noowners"),
                (
                    "Embedding Gemma Transcribe",
                    "exfat, local, nodev, nosuid, noowners, noatime, fskit",
                ),
            ],
        );
        for name in ["T7", "Studio", "Embedding Gemma Transcribe"] {
            assert_eq!(fake.validate(name), Ok(fake.volume(name)), "{name}");
        }
    }

    #[test]
    fn network_fat_and_read_only_volumes_are_refused_by_kind() {
        let fake = FakeVolumes::new(
            "refuse",
            &[
                ("media", "smbfs, nodev, nosuid, mounted by someone"),
                ("NO NAME", "msdos, local, nodev, nosuid, noowners, noatime"),
                (
                    "Installer",
                    "hfs, local, nodev, nosuid, read-only, noowners",
                ),
            ],
        );
        assert_eq!(fake.validate("media"), Err(LocationRefusal::Network));
        assert_eq!(
            fake.validate("NO NAME"),
            Err(LocationRefusal::SmallFileLimit {
                fs_type: "msdos".to_string()
            })
        );
        assert_eq!(fake.validate("Installer"), Err(LocationRefusal::ReadOnly));
    }

    #[test]
    fn the_mount_entry_is_matched_by_the_exact_volume_root_never_a_prefix() {
        // `T7` is listed last and read-only. A string-prefix match would take
        // it for a folder on `T7 Backup`.
        let fake = FakeVolumes::new(
            "exact",
            &[
                ("T7 Backup", "apfs, local, journaled"),
                ("T7", "hfs, local, read-only, journaled"),
            ],
        );
        assert_eq!(fake.validate("T7 Backup"), Ok(fake.volume("T7 Backup")));
        assert_eq!(fake.validate("T7"), Err(LocationRefusal::ReadOnly));

        // A volume with no entry of its own is unknown, even though an
        // ancestor folder has a writable local entry that contains it.
        std::fs::create_dir_all(fake.volume("Scratch").join("Models")).unwrap();
        let mut env = fake.env.clone();
        env.mount_source = MountSource::Fixed(vec![
            entry(fake.base.to_str().unwrap(), "apfs", true, false),
            entry(fake.volume("T7").to_str().unwrap(), "hfs", true, false),
        ]);
        assert_eq!(
            validate_location(&fake.volume("Scratch").join("Models"), &env),
            Err(LocationRefusal::UnknownVolume)
        );
    }

    #[test]
    fn non_directories_are_refused_before_the_disk_is_considered() {
        let fake = FakeVolumes::new("not-dir", &[("T7", "hfs, local, journaled")]);
        let file = fake.volume("T7").join("weights.bin");
        std::fs::write(&file, b"x").unwrap();
        assert_eq!(
            validate_location(&file, &fake.env),
            Err(LocationRefusal::NotADirectory)
        );
        assert_eq!(
            validate_location(&fake.volume("T7").join("missing"), &fake.env),
            Err(LocationRefusal::NotADirectory)
        );

        // Not a directory wins over same disk as home.
        let mut same_disk = fake.env.clone();
        same_disk.home_device = std::fs::metadata(&file).unwrap().dev();
        assert_eq!(
            validate_location(&file, &same_disk),
            Err(LocationRefusal::NotADirectory)
        );
    }

    #[test]
    fn a_mount_nested_inside_a_volume_is_not_judged_by_the_outer_volume() {
        // A share mounted at `<T7>/share` is lexically inside T7, so the
        // direct-child rule names T7 as its volume root. Tests cannot mount,
        // so a symlink to `/dev` (devfs, its own device) stands in for the
        // nested mount: inside T7 by path, on another device by `stat`.
        let fake = FakeVolumes::new("nested", &[("T7", "hfs, local, journaled")]);
        let t7 = fake.volume("T7");
        let share = t7.join("share");
        std::os::unix::fs::symlink("/dev", &share).unwrap();
        let share_device = std::fs::metadata(&share).unwrap().dev();
        let t7_device = std::fs::metadata(&t7).unwrap().dev();
        assert_ne!(share_device, t7_device, "/dev must be its own device");

        for (fs_type, local) in [("smbfs", false), ("msdos", true)] {
            let mut env = fake.env.clone();
            env.home_device = share_device.max(t7_device) + 1;
            env.mount_source = MountSource::Fixed(vec![
                entry(t7.to_str().unwrap(), "hfs", true, false),
                entry(share.to_str().unwrap(), fs_type, local, false),
            ]);
            // T7's writable local entry must not wave the share past the
            // network (1.4) and file-size (1.5) rules.
            assert_eq!(
                validate_location(&share, &env),
                Err(LocationRefusal::UnknownVolume),
                "{fs_type}"
            );
        }
    }

    #[test]
    fn a_later_mount_on_the_same_point_decides() {
        let fake = FakeVolumes::new("stacked", &[("T7", "hfs, local, journaled")]);
        let t7 = fake.volume("T7");
        let lower = entry(t7.to_str().unwrap(), "hfs", true, true);
        let upper = entry(t7.to_str().unwrap(), "apfs", true, false);
        let candidate = t7.join("Models");

        let mut env = fake.env.clone();
        env.mount_source = MountSource::Fixed(vec![lower.clone(), upper.clone()]);
        assert_eq!(validate_location(&candidate, &env), Ok(t7.clone()));
        env.mount_source = MountSource::Fixed(vec![upper, lower]);
        assert_eq!(
            validate_location(&candidate, &env),
            Err(LocationRefusal::ReadOnly)
        );
    }

    #[test]
    fn a_temp_folder_on_the_boot_disk_is_refused_as_same_disk_as_home_not_read_only() {
        let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is set"));
        let folder = scratch("boot-disk");
        let live = LocationEnv {
            mount_source: MountSource::System,
            volumes_root: PathBuf::from("/Volumes"),
            home_device: std::fs::metadata(&home).unwrap().dev(),
        };
        let reads = system_mount_reads();

        assert_eq!(
            validate_location(&folder, &live),
            Err(LocationRefusal::SameDiskAsHome)
        );
        // The device check alone decides an internal folder.
        assert_eq!(system_mount_reads(), reads);

        // Why the device check runs first: every internal path resolves to
        // the volume root `/`, whose entry is the sealed read-only system
        // volume. The writable Data volume is not matched by prefix.
        let other_home = LocationEnv {
            mount_source: MountSource::Fixed(parse_mount_table(
                "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n\
                 /dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled, nobrowse)\n",
            )),
            home_device: live.home_device.wrapping_add(1),
            ..live
        };
        assert_eq!(
            validate_location(&folder, &other_home),
            Err(LocationRefusal::ReadOnly)
        );

        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn building_the_system_env_and_the_default_store_never_reads_the_mount_table() {
        let reads = system_mount_reads();
        let store = Store::default_store().expect("HOME is set in tests");
        LocationEnv::system(store.root()).unwrap();
        assert_eq!(system_mount_reads(), reads);
    }

    #[test]
    fn the_system_source_reads_the_live_table_and_counts_the_read() {
        // The positive control for the no-spawn test: the counter moves when
        // the table really is read.
        let live = LocationEnv {
            mount_source: MountSource::System,
            volumes_root: PathBuf::from("/Volumes"),
            home_device: 0,
        };
        let reads = system_mount_reads();
        let table = live.mount_table().unwrap();
        assert_eq!(system_mount_reads(), reads + 1);
        assert!(
            table.iter().any(|e| e.mount_point == Path::new("/")),
            "{table:?}"
        );
    }

    #[test]
    fn the_system_env_is_volumes_and_the_home_device_without_overrides() {
        let base = scratch("system-env");
        // A home that does not exist yet takes the device it will be created
        // on, so a first run does not fail.
        let env = LocationEnv::system_with(&base.join("missing/.turbospark"), |_| None).unwrap();
        assert_eq!(
            env,
            LocationEnv {
                mount_source: MountSource::System,
                volumes_root: PathBuf::from("/Volumes"),
                home_device: std::fs::metadata(&base).unwrap().dev(),
            }
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    fn override_lookup(vars: HashMap<&'static str, OsString>) -> impl Fn(&str) -> Option<OsString> {
        move |name| vars.get(name).cloned()
    }

    #[test]
    fn empty_override_values_count_as_unset() {
        // Runs in both build kinds: release ignores the overrides, and debug
        // must treat a cleared variable as absent rather than fail on it.
        let base = scratch("empty-overrides");
        let vars = HashMap::from([
            ("TURBOSPARK_TEST_MOUNT_TABLE", OsString::new()),
            ("TURBOSPARK_TEST_VOLUMES_ROOT", OsString::new()),
            ("TURBOSPARK_TEST_HOME_DEVICE", OsString::new()),
        ]);
        let env = LocationEnv::system_with(&base, override_lookup(vars)).unwrap();
        assert_eq!(
            env,
            LocationEnv {
                mount_source: MountSource::System,
                volumes_root: PathBuf::from("/Volumes"),
                home_device: std::fs::metadata(&base).unwrap().dev(),
            }
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn debug_builds_honor_the_three_test_overrides() {
        let base = scratch("overrides");
        let table = base.join("mount.txt");
        std::fs::write(
            &table,
            "/dev/disk18s1 on /Volumes/T7 (hfs, local, journaled)\n",
        )
        .unwrap();
        let vars = HashMap::from([
            (
                "TURBOSPARK_TEST_MOUNT_TABLE",
                table.clone().into_os_string(),
            ),
            (
                "TURBOSPARK_TEST_VOLUMES_ROOT",
                base.join("Volumes").into_os_string(),
            ),
            ("TURBOSPARK_TEST_HOME_DEVICE", OsString::from("42")),
        ]);
        let env = LocationEnv::system_with(&base, override_lookup(vars.clone())).unwrap();
        assert_eq!(
            env,
            LocationEnv {
                mount_source: MountSource::Fixed(vec![entry("/Volumes/T7", "hfs", true, false)]),
                volumes_root: base.join("Volumes"),
                home_device: 42,
            }
        );

        let mut bad = vars;
        bad.insert("TURBOSPARK_TEST_HOME_DEVICE", OsString::from("disk3"));
        assert!(LocationEnv::system_with(&base, override_lookup(bad)).is_err());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn release_builds_ignore_the_test_overrides() {
        let base = scratch("overrides-release");
        let vars = HashMap::from([
            (
                "TURBOSPARK_TEST_MOUNT_TABLE",
                base.join("mount.txt").into_os_string(),
            ),
            (
                "TURBOSPARK_TEST_VOLUMES_ROOT",
                base.join("Volumes").into_os_string(),
            ),
            ("TURBOSPARK_TEST_HOME_DEVICE", OsString::from("42")),
        ]);
        let env = LocationEnv::system_with(&base, override_lookup(vars)).unwrap();
        assert_eq!(
            env,
            LocationEnv {
                mount_source: MountSource::System,
                volumes_root: PathBuf::from("/Volumes"),
                home_device: std::fs::metadata(&base).unwrap().dev(),
            }
        );
        let _ = std::fs::remove_dir_all(&base);
    }
}
