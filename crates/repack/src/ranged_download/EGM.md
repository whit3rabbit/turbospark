# Ranged Download

## Overview
Byte-range reads over a remote (HTTP) or in-memory source, used by the repacker to read safetensors and GGUF headers without downloading the tensor data, and to download whole files with resume via a range cache. `mod.rs` defines the `RangeSource` trait, the error type, and the header-fetch plans. `chunks.rs` partitions ranges and fills one buffer concurrently from disjoint slices. `http.rs` is the reqwest-blocking implementation with retry, pause/cancel, an on-disk range cache, and optional pinned-source validation.

## Interface
- `RangeSource` [mod.rs:RangeSource]: trait, `read_range(start, end_exclusive) -> Result<Vec<u8>, DownloadError>` by absolute offset
- `DownloadError` [mod.rs:DownloadError]: Request (also local filesystem failures and the refusing-to-overwrite check, not only transport errors [http.rs:PendingDownload::new, http.rs:PendingDownload::publish]), SourceIdentity, Cancelled, UnexpectedStatus{status, retry_after}, ShortRead, InvalidRange, Header, GgufHeader, TruncatedGguf
- `fetch_safetensors_header` [mod.rs:fetch_safetensors_header]: reads the 8-byte prefix, then exactly the declared header
- `fetch_gguf_header` [mod.rs:fetch_gguf_header]: geometric growth from `GGUF_INITIAL_FETCH_BYTES` (1 MiB) up to the GGUF cap, since GGUF has no length prefix
- `MemoryRangeSource` [mod.rs:MemoryRangeSource]: in-memory source for tests
- `HttpRangeSource` [http.rs:HttpRangeSource]: Range-GET source; builders `new`, `with_progress`, `with_cancel`, `with_token`, `with_optional_token`, `with_cache_dir`, `with_pinned_identity`
- `HttpRangeSource::download_to` / `download_batch` [http.rs:HttpRangeSource]: whole-file download (at most `FILE_DOWNLOAD_WORKERS` = 8 files per batch), published by rename
- `HttpRangeSource::invalidate_cache` / `source_cache_dir` [http.rs:HttpRangeSource]: per-source cache removal and location
- `CancelFlag` [http.rs:CancelFlag]: cloneable cancel plus pause/resume control (`cancel`, `pause`, `resume`, `is_paused`, `is_cancelled`, `checkpoint`, `wait_for_retry`, `same_flag`)
- `ByteProgressCallback` [http.rs:ByteProgressCallback]: per-chunk byte-count callback
- `hf_token_allowed_for_url` [http.rs:hf_token_allowed_for_url]: gate for sending the HF token
- `chunk_ranges`, `fill_chunks`, `MAX_RANGE_BYTES` (16 MiB), `RANGE_ATTEMPTS` (8), `RANGE_CONCURRENCY` (8) [chunks.rs] (crate-private to this module; `mod chunks` is not exported)

## Invariants
- A safetensors header length above `DEFAULT_MAX_HEADER_BYTES` is rejected before the second read, so a hostile prefix cannot size the allocation [mod.rs:fetch_safetensors_header]
- `MemoryRangeSource` returns `ShortRead` (never panics) for inverted or past-end ranges [mod.rs:MemoryRangeSource]
- `chunk_ranges` partitions the range exactly, in order, with no overlap, and terminates near `u64::MAX` (saturating add); it asserts `cap > 0` [chunks.rs:chunk_ranges]
- `fill_chunks` hands each worker a disjoint `&mut [u8]` (no unsafe), runs a single chunk on the calling thread, dispatches chunks in ascending index order and starts no new chunk once any has failed (in-flight ones finish), then reports the lowest-indexed failure deterministically [chunks.rs:fill_chunks]
- Only HTTP 206 is success; the client is `http1_only` so concurrent chunks use separate connections rather than one multiplexed HTTP/2 stream [http.rs:send_range, http.rs:build_client]
- Cancel is never retried; SourceIdentity errors are never retried; only 429/500/502/503/504 are throttled-retried (a delta-seconds Retry-After wins, capped at 30 s; the HTTP-date form is ignored by design and falls back to the doubling backoff); other statuses are fatal on the first attempt; transport errors (Request, ShortRead, ...) retry on 250 ms << attempt (shift capped at 4) [http.rs:with_retries, http.rs:throttled_status, http.rs:throttle_backoff, http.rs:send_range]
- A range past a known (or probed, for reads over one chunk) source length returns `ShortRead`; with unknown length, ranges over 64 GiB are `InvalidRange` [http.rs:check_range_within_source]
- With a pinned identity, each response must match the exact `Content-Range`, content length, ETag (when given), and end exactly at the expected byte, else `SourceIdentity`; `If-Match` is sent with the validator [http.rs:read_chunk, http.rs:send_range]
- With a pinned identity the cache key hashes the identity instead of the URL, so one pinned source shares a cache across URLs [http.rs:HttpRangeSource::cache_path]
- Cache entries are `<sha256 hex (64 bytes)> + bytes`; a wrong length or hash mismatch deletes the entry and refetches [http.rs:read_cached_chunk, http.rs:persist_cached_chunk]
- The bearer token is sent only to https `huggingface.co`/`hf.co` hosts and their subdomains, refusing userinfo, IPv6 literals, and lookalike hosts [http.rs:hf_token_allowed_for_url, http.rs:send_range]
- `download_batch` refuses to overwrite an existing destination (checked at creation and again at publish), fsyncs, and renames a complete file only; the partial file is removed on drop [http.rs:PendingDownload, http.rs:DownloadPartial]

## Side Effects
- Network: HTTP Range GETs (30 s connect, 600 s request timeout) [http.rs:build_client, http.rs:send_range]
- Filesystem: range cache files and `.range-<pid>-<n>.tmp` temps under the cache dir, written with create_new, fsync, rename [http.rs:persist_cached_chunk]
- Filesystem: creates parent dirs and a `.turbospark-download-<pid>-<n>.partial` file, then renames it to the destination [http.rs:PendingDownload]
- Filesystem: `invalidate_cache` removes the source's cache directory tree [http.rs:invalidate_cache]
- Threads: scoped worker threads per multi-chunk read and per download batch [chunks.rs:fill_chunks, http.rs:download_batch]
- Blocking: retry sleeps and pause waits via `thread::sleep` or the `CancelFlag` condvar [http.rs:wait_retry_delay]

## Verification
- Run: `cargo test -p turbospark-repack ranged_download`
