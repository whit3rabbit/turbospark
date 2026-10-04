//! Optional device-built pipeline archives. The hot cache remains address keyed;
//! disk identities are content keyed and include the compiler and device boundary.

use std::collections::HashMap;
use std::ffi::CStr;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use metal::{BinaryArchive, BinaryArchiveDescriptor, ComputePipelineDescriptor, Device, URL};

/// Cumulative host compilation timings, separate from GPU execution time.
/// A pipeline archive hit is counted only after Metal's fail-on-miss check succeeds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MetalCompilationStats {
    pub cache_enabled: bool,
    pub library_loads: u64,
    pub library_load_ms: f64,
    pub library_compiles: u64,
    pub library_compile_ms: f64,
    pub function_specializations: u64,
    pub function_specialization_ms: f64,
    pub pipeline_creations: u64,
    pub pipeline_create_ms: f64,
    pub in_memory_library_hits: u64,
    pub in_memory_function_hits: u64,
    pub in_memory_pipeline_hits: u64,
    pub archive_hits: u64,
    pub archive_misses: u64,
    pub archive_errors: u64,
    pub archive_load_ms: f64,
    pub archive_collect_ms: f64,
    pub archive_writes: u64,
    pub archive_write_ms: f64,
    pub buffer_allocations: u64,
    pub buffer_allocation_ms: f64,
}

pub(super) struct PipelineDiskCache {
    directory: PathBuf,
    identity: String,
    pending: HashMap<String, BinaryArchive>,
    source_digests: HashMap<usize, String>,
}

impl PipelineDiskCache {
    pub(super) fn from_environment(device: &Device) -> Option<Self> {
        if std::env::var("TURBOSPARK_METAL_PIPELINE_CACHE").as_deref() != Ok("1") {
            return None;
        }
        let directory = std::env::var_os("TURBOSPARK_METAL_CACHE_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join("Library/Caches/TurboSpark/metal"))
            })?;
        Self::at_directory(device, directory)
    }

    pub(super) fn at_directory(device: &Device, directory: PathBuf) -> Option<Self> {
        let version = sysctl_string(b"kern.osproductversion\0")?;
        // Specialized function archive serialization is supported from macOS 15.
        // macOS 14 keeps the existing source and pipeline compilation path.
        if !supports_specialized_archives(&version) {
            return None;
        }
        let build = sysctl_string(b"kern.osversion\0")?;
        let identity = format!(
            "turbospark-metal-archive-v1|{}|{}|{}|{}|{}",
            device.name(),
            device.registry_id(),
            version,
            build,
            std::env::consts::ARCH
        );
        let directory = if directory.is_absolute() {
            directory
        } else {
            std::env::current_dir().ok()?.join(directory)
        };
        Some(Self {
            directory,
            identity,
            pending: HashMap::new(),
            source_digests: HashMap::new(),
        })
    }

    pub(super) fn entry_key(
        &mut self,
        source: &'static str,
        function: &str,
        constants_key: &[u8],
        compiler_identity: &str,
    ) -> String {
        let source_digest = self
            .source_digests
            .entry(source.as_ptr() as usize)
            .or_insert_with(|| model_io::hash_data(source.as_bytes()));
        stable_entry_key(
            &self.identity,
            source_digest,
            function,
            constants_key,
            compiler_identity,
        )
    }

    pub(super) fn load(
        &self,
        device: &Device,
        key: &str,
        stats: &mut MetalCompilationStats,
    ) -> Option<BinaryArchive> {
        let started = Instant::now();
        let path = self.path(key);
        let result = if path.is_file() {
            let descriptor = BinaryArchiveDescriptor::new();
            descriptor.set_url(&file_url(&path));
            match device.new_binary_archive_with_descriptor(&descriptor) {
                Ok(archive) => Some(archive),
                Err(_) => {
                    stats.archive_errors += 1;
                    None
                }
            }
        } else {
            None
        };
        stats.archive_load_ms += started.elapsed().as_secs_f64() * 1000.0;
        result
    }

    pub(super) fn collect(
        &mut self,
        device: &Device,
        key: String,
        descriptor: &ComputePipelineDescriptor,
        stats: &mut MetalCompilationStats,
    ) {
        let started = Instant::now();
        let result = device.new_binary_archive_with_descriptor(&BinaryArchiveDescriptor::new());
        match result {
            Ok(archive) => match archive.add_compute_pipeline_functions_with_descriptor(descriptor)
            {
                Ok(true) => {
                    self.pending.insert(key, archive);
                }
                _ => stats.archive_errors += 1,
            },
            Err(_) => stats.archive_errors += 1,
        }
        stats.archive_collect_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    pub(super) fn flush(&mut self, stats: &mut MetalCompilationStats) {
        if self.pending.is_empty() {
            return;
        }
        let started = Instant::now();
        if std::fs::create_dir_all(&self.directory).is_err() {
            stats.archive_errors += self.pending.len() as u64;
            self.pending.clear();
        } else {
            // Each entry owns its archive. Concurrent writers for different
            // functions cannot lose each other's additions to a shared archive.
            for (key, archive) in self.pending.drain() {
                let destination = self.directory.join(format!("{key}.metalar"));
                match publish_archive(&archive, &destination) {
                    Ok(()) => stats.archive_writes += 1,
                    Err(_) => stats.archive_errors += 1,
                }
            }
        }
        stats.archive_write_ms += started.elapsed().as_secs_f64() * 1000.0;
    }

    fn path(&self, key: &str) -> PathBuf {
        self.directory.join(format!("{key}.metalar"))
    }
}

fn stable_entry_key(
    identity: &str,
    source_digest: &str,
    function: &str,
    constants_key: &[u8],
    compiler_identity: &str,
) -> String {
    let mut bytes = Vec::new();
    // Length framing prevents distinct component tuples from concatenating to
    // the same preimage, including arbitrary caller-supplied constant bytes.
    for value in [
        identity.as_bytes(),
        source_digest.as_bytes(),
        function.as_bytes(),
        constants_key,
        compiler_identity.as_bytes(),
    ] {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);
    }
    model_io::hash_data(&bytes)
}

fn publish_archive(archive: &BinaryArchive, destination: &Path) -> Result<(), String> {
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    let temporary = destination.with_extension(format!(
        "{}.{}.tmp",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    // Reserve an exclusive sibling before serialization. Rename publishes a
    // complete file atomically, even when another process writes the same key.
    let reservation = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| error.to_string())?;
    drop(reservation);
    let result = (|| {
        if !archive.serialize_to_url(&file_url(&temporary))? {
            return Err("Metal declined archive serialization".to_string());
        }
        std::fs::File::open(&temporary)
            .and_then(|file| file.sync_all())
            .map_err(|error| error.to_string())?;
        std::fs::rename(&temporary, destination).map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn file_url(path: &Path) -> URL {
    use std::os::unix::ffi::OsStrExt;
    let mut encoded = String::from("file://");
    for &byte in path.as_os_str().as_bytes() {
        if byte.is_ascii_alphanumeric() || b"/-._~".contains(&byte) {
            encoded.push(byte as char);
        } else {
            use std::fmt::Write;
            write!(encoded, "%{byte:02X}").expect("writing to a String cannot fail");
        }
    }
    let autoreleased = URL::new_with_string(&encoded);
    // metal-rs 0.33 wraps +URLWithString:'s autoreleased object as an owned
    // handle. Clone retains it; forgetting the original leaves its release to
    // the autorelease pool instead of releasing that reference twice.
    let retained = autoreleased.clone();
    std::mem::forget(autoreleased);
    retained
}

fn sysctl_string(name: &[u8]) -> Option<String> {
    let mut buffer = [0_u8; 256];
    let mut length = buffer.len();
    // SAFETY: name is a NUL-terminated static sysctl key; the output buffer
    // lives for the call and length describes its writable allocation.
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast(),
            buffer.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 || length == 0 || length > buffer.len() {
        return None;
    }
    CStr::from_bytes_with_nul(&buffer[..length])
        .ok()?
        .to_str()
        .ok()
        .map(str::to_owned)
}

fn supports_specialized_archives(version: &str) -> bool {
    version
        .split('.')
        .next()
        .and_then(|major| major.parse::<u32>().ok())
        .is_some_and(|major| major >= 15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_keys_cover_every_compilation_boundary() {
        let key = stable_entry_key("gpu-os-v1", "source", "fn", &[1, 2], "fast");
        assert_eq!(
            key,
            stable_entry_key("gpu-os-v1", "source", "fn", &[1, 2], "fast")
        );
        for other in [
            stable_entry_key("other-gpu-os", "source", "fn", &[1, 2], "fast"),
            stable_entry_key("gpu-os-v1", "other-source", "fn", &[1, 2], "fast"),
            stable_entry_key("gpu-os-v1", "source", "other-fn", &[1, 2], "fast"),
            stable_entry_key("gpu-os-v1", "source", "fn", &[1, 3], "fast"),
            stable_entry_key("gpu-os-v1", "source", "fn", &[1, 2], "precise"),
        ] {
            assert_ne!(key, other);
        }
        assert_ne!(
            stable_entry_key("a", "bc", "fn", &[], "fast"),
            stable_entry_key("ab", "c", "fn", &[], "fast")
        );
    }

    #[test]
    fn macos_14_uses_the_existing_pipeline_path() {
        assert!(!supports_specialized_archives("14.7.3"));
        assert!(supports_specialized_archives("15.0"));
        assert!(supports_specialized_archives("26.6.2"));
        assert!(!supports_specialized_archives("unknown"));
    }

    #[test]
    fn file_urls_escape_cache_directory_names() {
        super::super::autorelease_pool(|| {
            let url = file_url(Path::new("/tmp/a cache/#archive.metalar"));
            assert_eq!(url.path(), "/tmp/a cache/#archive.metalar");
        });
    }
}
