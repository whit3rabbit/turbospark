---
description: "Read expert blobs in place from a read-only mmap with MappedExpertLayer, and decide when that beats the pread slot cache."
icon: box
---

# Mapped experts

`MappedExpertLayer` maps one layer file read-only and returns the offset of each expert inside the mapping. Your code reads the expert bytes where the operating system already holds them, so no slot copy is made. Use it when the expert table fits in memory and you want to skip the copy. Use [`PreadExpertStreamer`](README.md) when memory is tight.

The choice is about who owns residency. A slot cache pins a bounded set of pages. A mapping leaves residency to the operating system, which can evict clean file-backed pages under memory pressure. Measurements behind this choice are in [expert residency](https://github.com/whit3rabbit/turbospark/blob/main/docs/EXPERT_RESIDENCY.md).

## Map a layer and read an expert

This example writes a file with a 100-byte header followed by four 64-byte blobs. It maps the window that starts after the header, prepares two experts, and checks the bytes at the offset `expert_offset` returns.

```rust
use std::fs::File;
use std::io::Write;

use turbospark_streaming::{MappedExpertLayer, MappedPagePreparationMode, StreamLayout};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("mapped-example.bin");
    let mut file = File::create(&path)?;
    file.write_all(&[0xFF; 100])?;
    for expert in 0..4u8 {
        file.write_all(&vec![expert; 64])?;
    }
    drop(file);

    let layout = StreamLayout {
        path: path.display().to_string(),
        stream_offset: 100,
        stream_size: 256,
        experts_per_layer: 4,
        expert_stride: 64,
        expert_offsets: None,
    };
    let layer = MappedExpertLayer::open(layout)?;

    let stats =
        layer.prepare_selected_experts(&[1, 3, 3], MappedPagePreparationMode::Touch, 1 << 20)?;
    println!("unique={} duplicates={}", stats.unique_experts, stats.duplicate_experts);

    let offset = layer.expert_offset(3)? as usize;
    let bytes = &layer.page_aligned_bytes()[offset..offset + layer.expert_stride() as usize];
    assert!(bytes.iter().all(|&b| b == 3));
    assert_eq!(layer.expert_bytes(3), Some(bytes));
    Ok(())
}
```

The output reports `unique=2 duplicates=1`. The duplicate index `3` is prepared once.

## Use offsets, not strides

`page_aligned_bytes()` starts on a page boundary. The window can start partway into that page, so the first expert is not at byte zero. Always get the position from `expert_offset(expert)`, which adds that shift, checks the expert index, and checks that the whole blob fits in the mapping. Do not compute `expert * expert_stride` yourself.

If you set `expert_offsets` in `StreamLayout`, supply one entry per expert. When the table is shorter than `experts_per_layer`, `expert_offset` falls back to the uniform `expert * stride` formula for the missing entries, while `prepare_selected_experts` returns an error. Keep the two lengths equal so both paths read the same bytes.

The mapping is owned by the `MappedExpertLayer`. Any pointer or wrapper you take from `page_aligned_bytes()` is valid only while that value is alive. The wrapper does not free the memory, so keep the layer alive for as long as the GPU or any reader uses it. This crate does not depend on Metal, so wrapping the bytes is your code's job.

## Prepare the experts you selected

`prepare_selected_experts` takes the expert indices for one routed batch and changes page access only. It never changes routing, bind order, or expert bytes.

| Argument | Meaning |
| --- | --- |
| `experts: &[usize]` | Expert indices for this batch. At most 64 per call. |
| `mode: MappedPagePreparationMode` | `Advice` asks the OS to fetch the ranges, and failures are counted but not returned. `Touch` reads one byte per VM page in this thread. |
| `max_bytes: u64` | Budget for the union of page spans. Whole experts are admitted in caller order while they fit, and a skipped expert does not block later smaller ones. A zero budget validates the indices and prepares nothing. |

A call with more than 64 experts returns an error. That error uses the `OpenFailed` variant, which does not describe the problem. Match on the call site rather than the variant.

The returned `MappedPagePreparationStats` counts mapped spans, not disk reads:

- `requested_experts`, `unique_experts`, and `duplicate_experts` describe the input.
- `prepared_experts` and `skipped_budget_experts` describe what the budget admitted.
- `selected_bytes` sums the logical size of the unique experts.
- `bytes_prepared` and `pages_prepared` count the union of the page spans that were admitted.
- `page_touches` counts the one-byte reads in `Touch` mode.
- `advisory_ranges` and `advisory_errors` count the `madvise` calls and their failures.
- `page_size` and `elapsed_ms` are the VM page size and the wall time of the call.

{% hint style="warning" %}
A prepared span is not a guarantee that the pages stay resident. The operating system can evict them after the call returns. Treat `bytes_prepared` as a statement about what this call touched, not about physical I/O.
{% endhint %}

## Other methods

- `prefetch_expert(expert) -> Result<u64, StreamerError>` touches each VM page of one expert and returns the touch count. It does not report disk bytes.
- `expert_bytes(expert) -> Option<&[u8]>` returns a host-side slice of one expert. Use it in tests and checks. The GPU path works from offsets.
- `expert_stride()`, `experts_per_layer()`, and `layout()` return the values from the `StreamLayout` this layer was opened with.
- `page_aligned_bytes()` returns the whole mapped buffer, starting on a page boundary.

## Errors

`MappedExpertLayer::open` returns `OpenFailed` when the file cannot be read or mapped, and `SizeMismatch` when the file is shorter than `stream_offset + stream_size`. `expert_offset`, `expert_bytes`, and `prefetch_expert` return `OffsetOutOfRange` for an index outside the layout or a blob outside the mapping. The full list is in the [crate overview](README.md#errors).

## Related

- [Crate overview](README.md) covers the platform table and the pread streamer.
- [Expert residency](https://github.com/whit3rabbit/turbospark/blob/main/docs/EXPERT_RESIDENCY.md) holds the measured trade between the two paths.
