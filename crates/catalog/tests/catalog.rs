//! The embedded catalog's structural invariants.
//!
//! Every assertion here is about the TABLE rather than about the network, so
//! this runs in the default suite. What it cannot see is whether the rows
//! still describe reality; `catalog_network.rs` is that, and it is
//! `#[ignore]`d.

use turbospark_catalog::{Catalog, SourceKind, Status};

#[test]
fn the_embedded_catalog_parses_and_every_row_validates() {
    let catalog = Catalog::embedded().expect("the embedded models.json parses");
    assert!(
        catalog.len() >= 10,
        "the curated table lost rows: {}",
        catalog.len()
    );
    for entry in catalog.entries() {
        entry
            .validate()
            .unwrap_or_else(|e| panic!("row {} does not validate: {e}", entry.alias));
    }
}

/// The family string is what `list` and `info` print and what a reader will
/// match against `docs/MODEL_FAMILY.md`. A typo there is invisible until
/// somebody greps for it, so it is pinned against the real enum.
#[test]
fn every_row_names_a_family_this_port_actually_has() {
    let known: Vec<&str> = model_io::all_known_architectures()
        .iter()
        .map(|a| a.family.as_str())
        .collect();
    for entry in Catalog::embedded().unwrap().entries() {
        assert!(
            known.contains(&entry.family.as_str()),
            "row {} claims family {:?}, which is not a ModelFamily here. Known: {known:?}",
            entry.alias,
            entry.family
        );
    }
}

/// **The rule this test exists for is the expensive one** (AGENTS.md
/// Gotcha 47): a GGUF carries llama.cpp's tokenizer representation, not an HF
/// `tokenizer.json`, so its sidecars MUST come from another repository. A row
/// that leaves `sidecars.repo` defaulted is claiming something impossible and
/// would 404 after the stream rather than before it.
#[test]
fn every_gguf_row_takes_its_sidecars_from_another_repo() {
    for entry in Catalog::embedded().unwrap().entries() {
        if entry.source.kind != SourceKind::Gguf {
            continue;
        }
        assert!(
            entry.sidecars.repo.is_some(),
            "gguf row {} must name a sidecar repo",
            entry.alias
        );
        assert_ne!(
            entry.sidecar_repo(),
            entry.source.repo,
            "gguf row {}'s sidecar repo is its own weights repo",
            entry.alias
        );
        assert!(
            entry.source.file.is_some(),
            "gguf row {} must name a file",
            entry.alias
        );
    }
}

/// An MLX row is the mirror: one repository holds both, and naming a file
/// would be meaningless because the walk reads the shard index.
#[test]
fn every_mlx_row_is_self_contained_and_names_no_file() {
    for entry in Catalog::embedded().unwrap().entries() {
        if entry.source.kind != SourceKind::Mlx {
            continue;
        }
        assert!(
            entry.source.file.is_none(),
            "mlx row {} names a file",
            entry.alias
        );
        assert_eq!(
            entry.sidecar_repo(),
            entry.source.repo,
            "mlx row {} takes sidecars from elsewhere, which is legal but has never \
             been needed -- if that changed, update this test rather than deleting it",
            entry.alias
        );
    }
}

/// **The sidecar list is per REPOSITORY and this asserts that it varies.**
/// `ternary27b` and `qwen38-27b` are ONE architecture and their file lists
/// are near-inverses: the first ships `merges.txt` and no
/// `generation_config.json`, the second the reverse. If this test ever goes
/// green because both lists became identical, somebody has copied one to the
/// other and a 20-minute stream is about to fail at the end.
#[test]
fn two_checkpoints_of_one_architecture_have_different_sidecar_lists() {
    let catalog = Catalog::embedded().unwrap();
    let ternary = catalog.get("ternary27b").expect("ternary27b row");
    let qwen38 = catalog.get("qwen38-27b").expect("qwen38-27b row");
    assert_eq!(ternary.family, qwen38.family, "these share an architecture");
    assert_ne!(
        ternary.sidecars.files, qwen38.sidecars.files,
        "the two lists are identical, which means one was copied from the other"
    );
    assert!(ternary.sidecars.files.iter().any(|f| f == "merges.txt"));
    assert!(!ternary
        .sidecars
        .files
        .iter()
        .any(|f| f == "generation_config.json"));
    assert!(!qwen38.sidecars.files.iter().any(|f| f == "merges.txt"));
    assert!(qwen38
        .sidecars
        .files
        .iter()
        .any(|f| f == "generation_config.json"));
}

/// **A GGUF install is not a compression step, so its `install_bytes` may not
/// sit below its `download_bytes`.**
///
/// The walk copies quantized expert bytes through VERBATIM and only transcodes
/// the small F32 resident core, so the two figures are equal in practice --
/// measured on the real TinyLlama pull, where a 904,385,920-byte GGUF produced
/// a 904,385,920-byte install. The field's job is the free-space warning,
/// where being generous is harmless and being short is not, and hand-estimated
/// figures drifted 6-8% UNDER on four rows before this test existed: on the
/// 26.9 GB Gemma Q8_0 row that is a two-gigabyte understatement of what a pull
/// needs.
///
/// MLX rows are deliberately exempt. They legitimately shrink, because the
/// walk drops the vision tower -- `bonsai27b` installs 3.9 GB from a 5.1 GB
/// checkpoint, and 333 vision tensors are 0.858 GiB of the difference.
///
/// **A GGUF WALK CAN SHRINK TOO, AND TWO ROWS NOW DO**, which is why an
/// `install_bytes` exactly equal to a `download_bytes` is a deliberate
/// rounding here rather than a copied figure. `ornith9b` really installs
/// 189,503 bytes UNDER its download (the F32-to-BF16 transcode of 177
/// tensors), and `ornith35b-gguf` 953 MB under (its source declares an MTP
/// block and `plan::classify` skips every `blk.<n>` at or above the trunk
/// count by name, so 256 experts never land). Neither is a reason to relax
/// this assertion: the field's contract is "approximate in the generous
/// direction only", so declaring the download size is both permitted and
/// conservative, and the measured truth is recorded in each row's `notes`.
/// Relaxing it would need a measured-install field the schema does not have,
/// and would give back the guard that caught four understated rows.
#[test]
fn a_gguf_rows_install_is_never_smaller_than_its_download() {
    for entry in Catalog::embedded().unwrap().entries() {
        if entry.source.kind != SourceKind::Gguf {
            continue;
        }
        assert!(
            entry.install_bytes >= entry.download_bytes,
            "{}: install_bytes {} is under download_bytes {}. A GGUF walk copies \
             expert bytes verbatim, so the install is at least as large; an \
             understated figure short-changes the free-space check.",
            entry.alias,
            entry.install_bytes,
            entry.download_bytes
        );
    }
}

/// A `verified` row claims a frozen number exists, so it has to name the test
/// that asserts it. Nothing else in the repo links the two.
#[test]
fn every_verified_row_names_at_least_one_gate() {
    for entry in Catalog::embedded().unwrap().entries() {
        if entry.status == Status::Verified {
            assert!(
                !entry.gates.is_empty(),
                "row {} claims `verified` and names no gate target",
                entry.alias
            );
        }
    }
}

/// A `caveat` row exists BECAUSE of its note, so an empty one is a row that
/// says "something is wrong" and does not say what.
#[test]
fn every_caveat_row_explains_itself() {
    for entry in Catalog::embedded().unwrap().entries() {
        if entry.status == Status::Caveat {
            let notes = entry
                .notes
                .as_deref()
                .unwrap_or_else(|| panic!("caveat row {} has no notes", entry.alias));
            assert!(notes.len() > 40, "row {}'s note is too thin", entry.alias);
        }
    }
}

/// A pinned row's revision must be a commit sha, because the whole value of
/// pinning is that the bytes cannot move under a frozen number. Rows at
/// `main` float on purpose (their publishers offer nothing else) and are
/// allowed; what is refused is a revision that is neither.
#[test]
fn a_revision_is_either_a_commit_sha_or_the_literal_main() {
    for entry in Catalog::embedded().unwrap().entries() {
        let rev = &entry.source.revision;
        let looks_like_sha = rev.len() == 40 && rev.chars().all(|c| c.is_ascii_hexdigit());
        assert!(
            rev == "main" || looks_like_sha,
            "row {}'s revision {rev:?} is neither `main` nor a 40-character sha",
            entry.alias
        );
    }
}

/// A user override merges by alias and is held to the same validation.
#[test]
fn a_user_override_replaces_a_row_and_is_validated() {
    let dir = std::env::temp_dir().join(format!("turbospark-catalog-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(
        dir.join("models.json"),
        r#"{"schema_version": 1, "models": [{
            "alias": "tinyllama",
            "name": "my own build",
            "family": "llama",
            "source": {"kind": "gguf", "repo": "me/mine", "revision": "main",
                       "file": "mine.gguf"},
            "sidecars": {"repo": "me/tokenizer", "revision": "main",
                         "files": ["tokenizer.json"]},
            "download_bytes": 1, "install_bytes": 1, "status": "runs"
        }]}"#,
    )
    .unwrap();
    let catalog = Catalog::load(&dir).expect("the override loads");
    assert_eq!(catalog.get("tinyllama").unwrap().name, "my own build");
    assert!(catalog.is_user_row("tinyllama"));
    assert!(
        !catalog.is_user_row("gemma4"),
        "an untouched curated row must not be marked as the user's"
    );

    // A malformed override fails at LOAD, which is the point: the same file
    // accepted here would 404 a quarter of an hour into a walk.
    std::fs::write(
        dir.join("models.json"),
        r#"{"schema_version": 1, "models": [{
            "alias": "broken",
            "name": "no sidecar repo on a gguf row",
            "family": "llama",
            "source": {"kind": "gguf", "repo": "me/mine", "revision": "main",
                       "file": "mine.gguf"},
            "sidecars": {"files": ["tokenizer.json"]},
            "download_bytes": 1, "install_bytes": 1, "status": "runs"
        }]}"#,
    )
    .unwrap();
    let err = Catalog::load(&dir).expect_err("a gguf row with no sidecar repo is refused");
    assert!(err.contains("sidecars.repo"), "unhelpful message: {err}");

    // A future schema is refused rather than best-effort parsed: the failure
    // mode of guessing is a silently dropped field.
    std::fs::write(
        dir.join("models.json"),
        r#"{"schema_version": 99, "models": []}"#,
    )
    .unwrap();
    let err = Catalog::load(&dir).expect_err("a future schema is refused");
    assert!(err.contains("schema_version"), "unhelpful message: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

/// **A row written before `measured` existed must still load.** The field is
/// additive and defaults, which is why the schema version did not move -- and
/// a user override on disk is exactly the file that was written against the
/// older shape. This is spelled out rather than left to
/// `a_user_override_replaces_a_row_and_is_validated` to imply, because that
/// test would keep passing if somebody made the field required and only
/// updated the fixture beside it.
#[test]
fn a_row_written_before_measured_existed_still_loads() {
    let dir = std::env::temp_dir().join(format!("turbospark-catalog-old-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("models.json"),
        r#"{"schema_version": 1, "models": [{
            "alias": "tinyllama",
            "name": "written before the measured field existed",
            "family": "llama",
            "source": {"kind": "gguf", "repo": "me/mine", "revision": "main",
                       "file": "mine.gguf"},
            "sidecars": {"repo": "me/tokenizer", "revision": "main",
                         "files": ["tokenizer.json"]},
            "download_bytes": 1, "install_bytes": 1, "status": "runs"
        }]}"#,
    )
    .unwrap();
    let catalog = Catalog::load(&dir).expect("a row with no measured block loads");
    let row = catalog.get("tinyllama").unwrap();
    assert!(row.measured.is_empty());
    assert!(
        row.measured_for("Apple M4 Max").is_none(),
        "no rows means no match, not a panic"
    );

    // And a malformed one fails at LOAD, like every other structural check
    // here. `min` above `max` is the specific mistake the worst-observed
    // convention invites: a bench footer prints the cases in protocol order,
    // not in speed order.
    std::fs::write(
        dir.join("models.json"),
        r#"{"schema_version": 1, "models": [{
            "alias": "backwards",
            "name": "min above max",
            "family": "llama",
            "source": {"kind": "gguf", "repo": "me/mine", "revision": "main",
                       "file": "mine.gguf"},
            "sidecars": {"repo": "me/tokenizer", "revision": "main",
                         "files": ["tokenizer.json"]},
            "download_bytes": 1, "install_bytes": 1, "status": "runs",
            "measured": [{"chip": "Apple M4 Max", "context": 4096,
                          "expert_cache_slots": 16, "peak_footprint_mib": 100,
                          "decode_tok_s_min": 40.0, "decode_tok_s_max": 20.0,
                          "measured_on": "2026-08-18", "source": "made up"}]
        }]}"#,
    )
    .unwrap();
    let err = Catalog::load(&dir).expect_err("a backwards measured row is refused");
    assert!(err.contains("above max"), "unhelpful message: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

/// **Every row whose gates include a memory oracle carries the evidence that
/// oracle was calibrated against.** This is the catalog half of the link;
/// `oracle_common::assert_agrees_with_catalog` is the other half and checks
/// the numbers agree. Without this one a row could quietly drop its measured
/// block and the oracle-side check would have nothing to compare against and
/// pass -- which is the shape of failure a paired test invites.
#[test]
fn every_row_with_a_memory_oracle_gate_records_what_that_oracle_measured() {
    let catalog = Catalog::embedded().expect("the embedded catalog parses");
    for entry in catalog.entries() {
        if !entry.gates.iter().any(|g| g.contains("memory_oracle")) {
            continue;
        }
        let m = entry
            .measured_for("Apple M4 Max")
            .unwrap_or_else(|| panic!("{}: names a memory oracle gate but records no measured row for the machine those gates were run on", entry.alias));
        assert!(
            m.context == 4096 || m.context == 8192,
            "{}: measured at {} context, which is neither protocol window",
            entry.alias,
            m.context
        );
    }
}

/// A missing override is not an error; the curated table is the whole answer.
#[test]
fn a_missing_user_override_is_not_an_error() {
    let dir = std::env::temp_dir().join(format!("turbospark-catalog-none-{}", std::process::id()));
    let loaded = Catalog::load(&dir).expect("no override is fine");
    assert_eq!(loaded.len(), Catalog::embedded().unwrap().len());
}

#[test]
fn find_matches_alias_name_and_repo() {
    let catalog = Catalog::embedded().unwrap();
    assert!(catalog.find("gemma").len() >= 3, "alias and name match");
    assert!(
        !catalog.find("mlx-community").is_empty(),
        "repo should be searchable"
    );
    assert!(catalog.find("no-such-model-anywhere").is_empty());
}
