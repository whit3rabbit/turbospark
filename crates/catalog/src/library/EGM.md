# catalog library

## Overview
Model libraries: places that hold model files apart from the TurboSpark home.
"This Mac" is the implicit internal library at `<home>/models`. Others are
folders on directly connected drives, listed in `<home>/libraries.json` and
described on the drive by a `library.json` marker so they survive renaming and
remounting. Drive contents are treated as untrusted.

## Interface
- `LibraryRecord`, `LibraryView`, `LibraryError`, `Availability`, `INTERNAL_LIBRARY_ID` [src/library/mod.rs:LibraryRecord, src/library/mod.rs:LibraryError, src/library/mod.rs:Availability, src/library/mod.rs:INTERNAL_LIBRARY_ID].
- Registry: `LibraryRegistry` (`load`, `libraries`, `add`, `remove`, `set_default`, `refresh`), `AddedLibrary`, `LoadReport`, `Refreshed`, `LIBRARIES_FILE` [src/library/registry.rs:LibraryRegistry].
- Marker: `LibraryMarker` (`read`, `write`, `from_installed`, `to_installed`), `PortableRow`, `MarkerError`, `MARKER_FILE`, `join_relative`, `new_library_id` [src/library/marker.rs:LibraryMarker, src/library/marker.rs:join_relative].
- Location: `locate`, `Located`, `discover_at`, `volume_relative_path`, `volume_root_of`, `STANDARD_LIBRARY_FOLDER` [src/library/locate.rs:locate].
- Mounts: `LocationEnv`, `MountEntry`, `MountSource`, `LocationRefusal`, `parse_mount_table`, `validate_location` [src/library/mount_table.rs:validate_location, src/library/mount_table.rs:LocationEnv].
- Transfer: `move_install`, `MoveError`, `MoveOutcome` [src/library/transfer.rs:move_install].
- Delete: `Store::delete_install`, `Store::delete_image_install` [src/library/delete.rs:Store::delete_install].
- Import: `ImportedStore`, and the `LibraryRegistry::import_moved_root` method [src/library/import.rs:import_moved_root].

## Invariants
- The internal library is never written into the `libraries` array, only its default flag [src/library/registry.rs: module docs].
- Every registry write re-reads the file just before its rename and applies only its own change [src/library/registry.rs: module docs].
- The marker is untrusted input: size capped, regular file only, ids/aliases/relative paths validated [src/library/marker.rs:MAX_MARKER_BYTES, src/library/marker.rs:validate_alias].
- Delete renames to a tombstone in the library's `.staging` before forgetting the row, and never deletes through the recorded path [src/library/delete.rs: module docs].
- A move copies, verifies, publishes by same-volume rename, repoints marker and registry, and only then deletes the source. Failure or cancel before the registry points at the copy removes the copy [src/library/transfer.rs: module docs].
- Links or special files in a move source refuse the move [src/library/transfer.rs: module docs, `MoveError`].
- Embedding and OCR models are copied into the home on import, never moved into a library [src/library/import.rs:HOME_ONLY_FOLDERS].
- Residual race between check and use is accepted (no unsafe, no openat) [src/library/delete.rs, src/library/transfer.rs: module docs].

## Side Effects
- Writes `<home>/libraries.json` and `library.json` markers [src/library/registry.rs:LIBRARIES_FILE, src/library/marker.rs:MARKER_FILE, src/library/marker.rs:replace_file].
- Copies and deletes install folders on disk [src/library/transfer.rs:move_install, src/library/delete.rs:Store::delete_install].
- Reads the OS mount table by spawning `/sbin/mount` [src/library/mount_table.rs:LocationEnv].

## Verification
- Run: `cargo test -p turbospark-catalog`
