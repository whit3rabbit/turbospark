---
description: "Read tensor entries from a .gturbo model_weights.bin with ResidentIndex and map the resident weight bytes read-only with ResidentBuffer."
icon: box
---
# Reading resident weights

Use `load_resident_index` to read the tensor table at the head of `model_weights.bin`. Then use `ResidentBuffer::map` to view the tensor bytes through a read-only memory map. The index is small, and the loader reads it fully into memory. The weight bytes stay on disk until your code touches them.

The one detail that trips people up is the offset base. Entry offsets are absolute file offsets, so you subtract `index_size` before you slice the buffer.

## File layout

`model_weights.bin` has two regions. The index region comes first, and the resident tensor data follows it:

```text
+-----------------------------------+ 0
| header (24 bytes)                 |
| entry table (72 bytes per entry)  |  index region: index_size bytes
| name string table, page padding   |
+-----------------------------------+ index_size
| resident tensor data              |  resident region: resident_size bytes
+-----------------------------------+ index_size + resident_size
```

`ResidentIndexHeader` carries three values. `index_size` is the full length of the index region, including the header, the table, the names, and the padding. `resident_size` is the length of the tensor data. `entry_count` is the number of entries.

Each `ResidentIndexEntry` names one tensor. It carries a `dtype` tag and the absolute `file_offset` and `size_bytes` of its payload. It also carries four `shape` dimensions. The `scale` and `bias` planes each have their own offset and size.

`load_resident_index` checks every offset before it returns. Payload, scale, and bias ranges must fall inside the resident region. Duplicate names are rejected. Dtypes 4 and 5 are treated as fixed affine tensors (4-bit and 8-bit), so their column count, plane sizes, and companion alignment are checked too.

## Read the index and map the weights

This example prints each tensor's name, dtype, and payload length. It compiles against the crate and runs against a small fixture file.

```rust
use std::path::Path;

use turbospark_model_io::{load_resident_index, ResidentBuffer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: resident <model_weights.bin>");
    let weights = Path::new(&path);

    let index = load_resident_index(weights)?;
    let buffer = ResidentBuffer::map(weights, index.header.index_size, index.header.resident_size)?;

    let mut total = 0_u64;
    for entry in index.entries.values() {
        if entry.size_bytes == 0 {
            continue;
        }
        let start = (entry.file_offset - index.header.index_size) as usize;
        let end = start + entry.size_bytes as usize;
        let payload = &buffer.data()[start..end];
        total += payload.len() as u64;
        println!(
            "{} dtype={} bytes={}",
            entry.name,
            entry.dtype,
            payload.len()
        );
    }
    println!("{} tensors, {total} payload bytes", index.entries.len());
    Ok(())
}
```

Running it on a fixture with one 64-byte tensor named `w` prints (this audit did not reproduce that fixture):

```text
w dtype=0 bytes=64
1 tensors, 64 payload bytes
```

Two details matter in that code:

* `ResidentBuffer::map` starts the view at `index_size`, so `buffer.data()[0]` is file byte `index_size`. Subtract `index_size` from each `file_offset` to get an index into `data()`.
* Zero-size entries pass validation with any offset, including one below `index_size`. Skip them before you subtract, or the subtraction can underflow.

`index.entries` is a `HashMap<String, ResidentIndexEntry>`. For a single tensor, call `index.entries.get("name")` instead of iterating. Iteration order is not defined.

## The mapped buffer

`ResidentBuffer::map(file_path, file_offset, resident_size)` maps the bytes from `file_offset` through `file_offset + resident_size`. Four methods expose the result:

| Method | Returns |
| --- | --- |
| `data()` | The `resident_size` bytes, starting at `file_offset` |
| `mapped_bytes()` | The whole mapping, starting at the page boundary at or below `file_offset` |
| `slice_shift()` | The distance from the start of `mapped_bytes()` to the start of `data()`. Zero when `file_offset` is page-aligned |

`map` first checks that the region ends inside the file. A truncated file returns `ModelError::TensorSizeMismatch`, which names the file and both lengths. Without that check, the first read of a truncated mapping would crash the process with `SIGBUS`.

The page size comes from `sysconf(_SC_PAGESIZE)` at runtime, so the crate handles 16 KiB pages on Apple Silicon. The mapping is read-only.

> The buffer assumes the file does not change while it is mapped. Do not truncate or rewrite an install while a `ResidentBuffer` over it is alive. The crate does not guard against outside writers.

The `mmap` call and the page-size query into `libc` are the two `unsafe` blocks on this path.

### Handing the bytes to a GPU

A GPU backend that needs a page-aligned base address can wrap `mapped_bytes()` without a copy. It then adds `slice_shift()` to every tensor offset. The Metal wrapping itself lives in the `gpu` crate in this workspace, not in this crate.

## Related

* [Overview](README.md)
* [Validating an install](validating-an-install.md)
* [docs.rs: ResidentBuffer](https://docs.rs/turbospark-model-io/latest/turbospark_model_io/struct.ResidentBuffer.html)
* [GTurbo format](https://github.com/whit3rabbit/turbospark/blob/main/docs/GTURBO.md)
