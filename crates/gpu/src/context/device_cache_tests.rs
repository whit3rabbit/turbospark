//! Real Metal archive reuse and fallback tests. No process-wide environment
//! mutation, so default and opt-in paths can be checked concurrently.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use metal::{FunctionConstantValues, MTLDataType};

use super::*;
use crate::context::{autorelease_pool, dispatch_threads_3d, read_buffer_bytes};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
constant uint VALUE [[function_constant(0)]];
kernel void cache_test(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = VALUE + tid;
}
kernel void other_test(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = VALUE + 10 + tid;
}
"#;

const CHANGED_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
constant uint VALUE [[function_constant(0)]];
kernel void cache_test(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = VALUE + 100 + tid;
}
"#;

struct CacheDirectory(PathBuf);

impl CacheDirectory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "turbospark metal cache {} {}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn files(&self) -> Vec<PathBuf> {
        std::fs::read_dir(&self.0)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect()
    }
}

impl Drop for CacheDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn context(directory: Option<&Path>, precise_math: bool) -> MetalContext {
    let device = Device::system_default().expect("Metal device required");
    let cache =
        directory.and_then(|path| PipelineDiskCache::at_directory(&device, path.to_path_buf()));
    MetalContext::with_options(device, cache, precise_math, false)
}

fn run(
    context: &mut MetalContext,
    source: &'static str,
    function: &'static str,
    value: u32,
) -> Vec<u32> {
    let constants = FunctionConstantValues::new();
    constants.set_constant_value_at_index((&value as *const u32).cast(), MTLDataType::UInt, 0);
    let pipeline = context
        .pipeline(source, function, &constants, &value.to_le_bytes())
        .unwrap();
    let buffer = context.new_output_buffer(16);
    dispatch_threads_3d(
        context,
        &pipeline,
        &[(&buffer, 0)],
        &[],
        (4, 1, 1),
        (4, 1, 1),
    );
    read_buffer_bytes(&buffer, 0, 16)
        .chunks_exact(4)
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
fn pipeline_cache_reuses_real_specializations_in_second_context() {
    autorelease_pool(|| {
        let directory = CacheDirectory::new();
        let mut first = context(Some(&directory.0), false);
        if !first.compilation_stats().cache_enabled {
            eprintln!("specialized archive test requires macOS 15+");
            return;
        }
        assert_eq!(run(&mut first, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        assert_eq!(run(&mut first, SOURCE, "cache_test", 17), [17, 18, 19, 20]);
        assert_eq!(run(&mut first, SOURCE, "other_test", 7), [17, 18, 19, 20]);
        assert_eq!(first.compilation_stats().archive_misses, 3);
        assert!(
            directory.files().is_empty(),
            "flush must stay off the dispatch path"
        );
        first.flush_pipeline_cache();
        assert_eq!(first.compilation_stats().archive_writes, 3);
        assert_eq!(first.compilation_stats().archive_errors, 0);
        assert_eq!(directory.files().len(), 3);
        drop(first);

        let mut second = context(Some(&directory.0), false);
        assert_eq!(run(&mut second, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        assert_eq!(run(&mut second, SOURCE, "cache_test", 17), [17, 18, 19, 20]);
        assert_eq!(run(&mut second, SOURCE, "other_test", 7), [17, 18, 19, 20]);
        assert_eq!(second.compilation_stats().archive_hits, 3);
        assert_eq!(second.compilation_stats().archive_misses, 0);
        assert_eq!(second.compilation_stats().archive_errors, 0);
        assert_eq!(second.compilation_stats().in_memory_library_hits, 2);
        assert_eq!(run(&mut second, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        assert_eq!(second.compilation_stats().in_memory_pipeline_hits, 1);
        second.flush_pipeline_cache();
        assert_eq!(second.compilation_stats().archive_writes, 0);
    });
}

#[test]
fn pipeline_cache_invalidates_changed_source_constants_and_math() {
    autorelease_pool(|| {
        let directory = CacheDirectory::new();
        let mut first = context(Some(&directory.0), false);
        if !first.compilation_stats().cache_enabled {
            return;
        }
        assert_eq!(run(&mut first, SOURCE, "cache_test", 3), [3, 4, 5, 6]);
        first.flush_pipeline_cache();
        drop(first);

        let mut changed = context(Some(&directory.0), false);
        assert_eq!(
            run(&mut changed, CHANGED_SOURCE, "cache_test", 3),
            [103, 104, 105, 106]
        );
        assert_eq!(
            run(&mut changed, SOURCE, "cache_test", 13),
            [13, 14, 15, 16]
        );
        assert_eq!(changed.compilation_stats().archive_hits, 0);
        assert_eq!(changed.compilation_stats().archive_misses, 2);
        changed.flush_pipeline_cache();

        let mut precise = context(Some(&directory.0), true);
        assert_eq!(run(&mut precise, SOURCE, "cache_test", 3), [3, 4, 5, 6]);
        assert_eq!(precise.compilation_stats().archive_hits, 0);
        assert_eq!(precise.compilation_stats().archive_misses, 1);
        precise.flush_pipeline_cache();
        assert_eq!(directory.files().len(), 4);
    });
}

#[test]
fn pipeline_cache_corrupt_and_missing_archive_fall_back_and_repair() {
    autorelease_pool(|| {
        let directory = CacheDirectory::new();
        let mut first = context(Some(&directory.0), false);
        if !first.compilation_stats().cache_enabled {
            return;
        }
        assert_eq!(run(&mut first, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        first.flush_pipeline_cache();
        drop(first);
        let path = directory.files().pop().unwrap();
        std::fs::write(&path, b"invalid Metal archive").unwrap();

        let mut repair = context(Some(&directory.0), false);
        assert_eq!(run(&mut repair, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        assert_eq!(repair.compilation_stats().archive_hits, 0);
        assert_eq!(repair.compilation_stats().archive_misses, 1);
        assert_eq!(repair.compilation_stats().archive_errors, 1);
        repair.flush_pipeline_cache();
        assert_eq!(repair.compilation_stats().archive_writes, 1);
        drop(repair);

        let mut restored = context(Some(&directory.0), false);
        assert_eq!(run(&mut restored, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        assert_eq!(restored.compilation_stats().archive_hits, 1);
        drop(restored);
        std::fs::remove_file(path).unwrap();

        let mut missing = context(Some(&directory.0), false);
        assert_eq!(run(&mut missing, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        assert_eq!(missing.compilation_stats().archive_hits, 0);
        assert_eq!(missing.compilation_stats().archive_misses, 1);
    });
}

#[test]
fn pipeline_cache_valid_archive_with_wrong_specialization_is_a_miss() {
    autorelease_pool(|| {
        let directory = CacheDirectory::new();
        let mut first = context(Some(&directory.0), false);
        if !first.compilation_stats().cache_enabled {
            return;
        }
        assert_eq!(run(&mut first, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        first.flush_pipeline_cache();
        let original = directory.files().pop().unwrap();
        let compiler = first.compiler_identities[&(SOURCE.as_ptr() as usize)].clone();
        let wrong_key = first.disk_cache.as_mut().unwrap().entry_key(
            SOURCE,
            "cache_test",
            &17_u32.to_le_bytes(),
            &compiler,
        );
        std::fs::copy(original, directory.0.join(format!("{wrong_key}.metalar"))).unwrap();
        drop(first);

        let mut second = context(Some(&directory.0), false);
        assert_eq!(run(&mut second, SOURCE, "cache_test", 17), [17, 18, 19, 20]);
        assert_eq!(second.compilation_stats().archive_hits, 0);
        assert_eq!(second.compilation_stats().archive_misses, 1);
        assert_eq!(second.compilation_stats().archive_errors, 0);
        second.flush_pipeline_cache();
        assert_eq!(second.compilation_stats().archive_writes, 1);
        drop(second);

        let mut repaired = context(Some(&directory.0), false);
        assert_eq!(
            run(&mut repaired, SOURCE, "cache_test", 17),
            [17, 18, 19, 20]
        );
        assert_eq!(repaired.compilation_stats().archive_hits, 1);
    });
}

#[test]
fn pipeline_cache_disabled_and_unwritable_paths_keep_outputs() {
    autorelease_pool(|| {
        let directory = CacheDirectory::new();
        let mut disabled = context(None, false);
        assert_eq!(run(&mut disabled, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        disabled.flush_pipeline_cache();
        assert!(!disabled.compilation_stats().cache_enabled);
        assert_eq!(disabled.compilation_stats().archive_misses, 0);
        assert_eq!(disabled.compilation_stats().archive_writes, 0);
        assert_eq!(disabled.compilation_stats().library_compiles, 1);
        assert_eq!(disabled.compilation_stats().function_specializations, 1);
        assert_eq!(disabled.compilation_stats().pipeline_creations, 1);
        assert_eq!(disabled.compilation_stats().buffer_allocations, 1);
        assert!(disabled.compilation_stats().buffer_allocation_ms > 0.0);

        let blocked = directory.0.join("file-as-directory");
        std::fs::write(&blocked, b"blocking cache directory").unwrap();
        let mut unwritable = context(Some(&blocked), false);
        assert_eq!(run(&mut unwritable, SOURCE, "cache_test", 7), [7, 8, 9, 10]);
        unwritable.flush_pipeline_cache();
        if unwritable.compilation_stats().cache_enabled {
            assert_eq!(unwritable.compilation_stats().archive_errors, 1);
            assert_eq!(unwritable.compilation_stats().archive_writes, 0);
        }
    });
}

#[test]
fn pipeline_cache_concurrent_writers_publish_complete_entries() {
    let directory = CacheDirectory::new();
    if !context(Some(&directory.0), false)
        .compilation_stats()
        .cache_enabled
    {
        return;
    }
    let workers: Vec<_> = [7, 17]
        .into_iter()
        .map(|value| {
            let path = directory.0.clone();
            std::thread::spawn(move || {
                autorelease_pool(|| {
                    let mut context = context(Some(&path), false);
                    assert_eq!(run(&mut context, SOURCE, "cache_test", value)[0], value);
                    // Both workers also write the same key concurrently.
                    assert_eq!(run(&mut context, SOURCE, "other_test", 7)[0], 17);
                    context.flush_pipeline_cache();
                    assert_eq!(context.compilation_stats().archive_writes, 2);
                    assert_eq!(context.compilation_stats().archive_errors, 0);
                });
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(directory.files().len(), 3);
    autorelease_pool(|| {
        let mut reader = context(Some(&directory.0), false);
        assert_eq!(run(&mut reader, SOURCE, "cache_test", 7)[0], 7);
        assert_eq!(run(&mut reader, SOURCE, "cache_test", 17)[0], 17);
        assert_eq!(run(&mut reader, SOURCE, "other_test", 7)[0], 17);
        assert_eq!(reader.compilation_stats().archive_hits, 3);
        assert_eq!(reader.compilation_stats().archive_errors, 0);
    });
}

#[test]
fn pipeline_inventory_lists_each_prepared_specialization_once_in_sorted_order() {
    autorelease_pool(|| {
        // No disk directory: the inventory reports the in-memory pipeline
        // map, which exists on every Metal-capable macOS version.
        let mut context = context(None, false);
        assert!(
            context.pipeline_inventory().is_empty(),
            "a fresh context has prepared nothing"
        );

        assert_eq!(run(&mut context, SOURCE, "cache_test", 7)[0], 7);
        assert_eq!(run(&mut context, SOURCE, "cache_test", 17)[0], 17);
        assert_eq!(run(&mut context, SOURCE, "other_test", 7)[0], 17);
        // An in-memory hit must not add a duplicate inventory entry.
        assert_eq!(run(&mut context, SOURCE, "cache_test", 7)[0], 7);

        let expected: Vec<(String, Vec<u8>)> = [
            ("cache_test", 7u32),
            ("cache_test", 17u32),
            ("other_test", 7u32),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_string(), value.to_le_bytes().to_vec()))
        .collect();
        assert_eq!(context.pipeline_inventory(), expected);
        // The read is diagnostic: repeating it changes no dispatch accounting.
        assert_eq!(context.pipeline_inventory(), expected);
        assert_eq!(context.compilation_stats().in_memory_pipeline_hits, 1);
    });
}
