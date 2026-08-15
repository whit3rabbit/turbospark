//! Printing. No decisions.
//!
//! The one editorial choice in here is what goes ABOVE the fold on a probe:
//! the verdict, then the numbers that would change a reader's mind about a
//! download, then everything else. The expert-cache arithmetic is high on
//! that list even though it gates nothing, because it is the number that
//! decides whether a model FITS on this engine and it is invisible in every
//! other listing -- Mixtral is correct here and wants 54.5 GiB of pinned slot
//! cache, and the whole point of printing it is that nobody discovers that
//! after a 26 GB download twice.

use catalog::{CatalogEntry, ProbeReport, Store, Verdict};

pub use catalog::human_bytes;

/// One catalog row in full.
pub fn entry(entry: &CatalogEntry, is_user_row: bool, store: &Store) {
    println!("{}  ({})", entry.alias, entry.name);
    println!("  family      {}", entry.family);
    println!("  status      {}", entry.status.as_str());
    println!(
        "  source      {} @ {} [{}]",
        entry.source.repo,
        entry.source.revision,
        entry.source.kind.as_str()
    );
    if let Some(file) = &entry.source.file {
        println!("  file        {file}");
    }
    println!(
        "  sidecars    {} @ {}: {}",
        entry.sidecar_repo(),
        entry.sidecar_revision(),
        entry.sidecars.files.join(", ")
    );
    println!(
        "  size        {} to download, about {} installed",
        human_bytes(entry.download_bytes),
        human_bytes(entry.install_bytes)
    );
    if !entry.gates.is_empty() {
        println!("  gates       {}", entry.gates.join(", "));
    }
    if is_user_row {
        println!("  origin      your models.json, not the curated table");
    }
    if entry.source.revision == "main" {
        // Worth saying out loud: a floating row's frozen numbers stop
        // meaning anything the moment the publisher re-uploads, and the size
        // above is the only thing that would notice.
        println!(
            "  note        pinned at `main`, so this row FLOATS. Its size is the \
             only fingerprint."
        );
    }
    match store.installed().get(&entry.alias) {
        Some(row) => println!(
            "  installed   {} on {} ({})",
            row.path.display(),
            row.installed_on,
            human_bytes(row.install_bytes)
        ),
        None => println!(
            "  installed   no -- `turbospark-model pull {}`",
            entry.alias
        ),
    }
    if let Some(notes) = &entry.notes {
        println!("\n{notes}");
    }
}

/// A probe report.
pub fn report(report: &ProbeReport) {
    println!();
    match &report.verdict {
        Verdict::Runnable => println!("RUNNABLE  {}", report.repo),
        Verdict::Refused(why) => println!("REFUSED   {}\n          {why}", report.repo),
    }

    if let Some(file) = &report.file {
        println!("  file        {file}");
    }
    if let Some(arch) = &report.architecture {
        println!("  declares    {arch}");
    }
    if let Some(family) = report.family {
        println!("  family      {}", family.as_str());
    }
    if let Some(bytes) = report.download_bytes {
        println!("  download    {}", human_bytes(bytes));
    }
    if let Some((bits, group)) = report.affine {
        println!("  quant       MLX affine, {bits}-bit at group {group}");
    }

    if let Some(arch) = &report.arch {
        println!(
            "  shape       {} layers, {} hidden, {} vocab, {}",
            arch.num_layers,
            arch.hidden_size,
            arch.vocab_size,
            if arch.num_experts > 0 {
                format!("{} experts top-{}", arch.num_experts, arch.top_k_experts)
            } else {
                "dense".to_string()
            }
        );
    }

    if !report.types.is_empty() {
        println!("  block types");
        let total: u64 = report.types.iter().filter_map(|t| t.bytes).sum();
        for share in &report.types {
            let bytes = match share.bytes {
                // UNSIZED, never rendered as zero: a type this port cannot
                // size sorts to the bottom of a share column as 0.0%, which
                // is the exact inverse of its real rank, and on a mixed file
                // it is usually the routed experts.
                None => "UNSIZED".to_string(),
                Some(b) if total > 0 => {
                    format!(
                        "{} ({:.1}%)",
                        human_bytes(b),
                        100.0 * b as f64 / total as f64
                    )
                }
                Some(b) => human_bytes(b),
            };
            println!(
                "    {:<9} {:>4} tensors  {:<20} {}",
                share.name,
                share.tensors,
                bytes,
                match (share.executable, share.transcoded) {
                    // Not "has kernels": F32/F16/BF16 reach no dispatch at
                    // all, they are narrowed at repack. Saying otherwise
                    // sends a reader looking for an F32 GEMV.
                    (_, true) => "transcoded at repack",
                    (true, false) => "has kernels",
                    (false, false) => "NO KERNELS",
                }
            );
        }
    }

    // The number that decides whether a model fits here. See the module
    // header for why it is printed even though it gates nothing.
    if let Some(stride) = report.expert_stride {
        println!(
            "  experts     one expert is {}, so the pinned slot cache would be",
            human_bytes(stride)
        );
        for (slots, bytes) in report.slot_cache_bytes() {
            let verdict = if bytes > 8 * 1024 * 1024 * 1024 {
                "  <-- will not fit"
            } else {
                ""
            };
            println!(
                "                {slots:>2} slots: {}{verdict}",
                human_bytes(bytes)
            );
        }
        println!(
            "              (slots x layers x expert stride. Expert GRANULARITY \
             decides what this engine can hold, not model size.)"
        );
    }

    println!(
        "  tokenizer   {}",
        if report.sidecars_present.is_empty() {
            "none found".to_string()
        } else {
            report.sidecars_present.join(", ")
        }
    );
    match &report.chat_template {
        Some(where_) => println!("  template    {where_}"),
        None => println!("  template    NONE FOUND"),
    }
    if !report.sidecars_missing.is_empty() {
        println!("  absent      {}", report.sidecars_missing.join(", "));
    }
    for warning in &report.warnings {
        println!("  ! {warning}");
    }
    println!();
}
