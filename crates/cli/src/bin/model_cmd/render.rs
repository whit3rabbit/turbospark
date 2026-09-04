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
    if let Some(mtp) = &entry.mtp {
        println!(
            "  mtp head    {} @ {} (source's own conversion has none)",
            mtp.repo, mtp.revision
        );
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

/// The ranked table, then the rows that need explaining.
///
/// **The two size columns are the whole point and they are not the same
/// question.** ALLOCS is what the engine allocates -- the expert-cache slots
/// and the KV -- so exceeding memory there is a failed open. ON DISK is the
/// whole install, and exceeding memory there is the streaming this engine is
/// built around. A single "size" column would either call a 13 GB install on
/// a 16 GB machine impossible (it runs) or call a 27 GB one comfortable (it
/// thrashes).
///
/// The `*` and `~` prefixes on ALLOCS are load-bearing rather than
/// decoration: `*` is a measurement from `models.json` and `~` an estimate
/// from the checkpoint's shape, and a reader who cannot tell them apart
/// cannot tell which numbers a test can go red over.
pub fn recommendations(rows: &[catalog::Recommendation], machine: &catalog::Machine, context: u32) {
    println!(
        "machine: {} of memory{}{}",
        human_bytes(machine.physical_bytes),
        if machine.chip.is_empty() {
            String::new()
        } else {
            format!(" on {}", machine.chip)
        },
        match machine.working_set_bytes {
            Some(ws) => format!(", {} of Metal working set", human_bytes(ws)),
            None => String::new(),
        }
    );
    // Named only when it is NOT the default, so the common header stays the
    // line every existing note quotes. Reported at all because the tier moves
    // the VERDICTS: `strict` on a 10 GiB machine drops a row that `relaxed`
    // reports as fitting, and nothing else in this output says why.
    let guard = match machine.load_guard {
        model_io::LoadGuard::Relaxed => String::new(),
        model_io::LoadGuard::Custom { max_counted_bytes } => format!(
            ", load guard custom ({} ceiling)",
            human_bytes(max_counted_bytes)
        ),
        other => format!(", load guard {}", other.as_str()),
    };
    println!("fitting against a {context}-token context{guard}\n");

    if rows.is_empty() {
        println!("nothing to rank");
        return;
    }

    let width = rows
        .iter()
        .map(|r| r.origin.install_target().len())
        .max()
        .unwrap_or(8)
        .clamp(8, 40);
    println!(
        "  {:<width$}  {:<9}  {:>10}  {:>9}  {:>9}  VERDICT",
        "MODEL",
        "EVIDENCE",
        "ALLOCS",
        "ON DISK",
        "TOK/S",
        width = width
    );
    for row in rows {
        let target = row.origin.install_target();
        println!(
            "  {:<width$}  {:<9}  {:>10}  {:>9}  {:>9}  {}",
            truncate(&target, width),
            row.evidence.as_str(),
            counted_column(&row.fit),
            human_bytes(row.fit.mapped),
            tok_s_column(row),
            verdict_column(row),
            width = width
        );
    }

    println!(
        "\nallocs: what the engine allocates and phys_footprint charges for \
         (expert-cache slots + KV). on disk: the whole install, weights included -- \
         those STREAM, so they need not fit."
    );
    println!(
        "tok/s: measured on this chip, never estimated -- decode rate does not track \
         weight bytes here. A dash means nobody has measured it."
    );
    println!("allocs: * measured, ~ estimated from the checkpoint's shape, ? not read yet.");

    let unknowns = rows
        .iter()
        .filter(|r| r.fit.counted_source == catalog::CountedSource::Unknown)
        .count();
    if unknowns > 0 {
        println!(
            "\n{unknowns} row(s) report `unknown`: nothing has read their headers, so the \
             slot cache and the KV cannot be computed. `recommend --probe` reads them \
             (one header per row) and `probe <repo>` reads one."
        );
    }

    print_notes(rows);
}

/// Per-row notes, with anything said about more than one row lifted into a
/// single line.
///
/// Without this the common caveats drown the table they annotate: on this
/// machine every measured row carries the same sentence about the slot count
/// its peak was taken at, which is worth reading ONCE and is eight paragraphs
/// of noise repeated per row.
fn print_notes(rows: &[catalog::Recommendation]) {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for row in rows {
        for note in &row.notes {
            *counts.entry(note.as_str()).or_default() += 1;
        }
    }
    let shared: Vec<&str> = counts
        .iter()
        .filter(|(_, &n)| n > 1)
        .map(|(note, _)| *note)
        .collect();
    if !shared.is_empty() {
        println!();
        for note in &shared {
            // The COUNT, not "all rows": these notes are shared by several
            // and true of none of the others, and a reader who takes an
            // architecture refusal as applying to the whole table has been
            // told something false.
            println!("{} rows: {note}", counts[note]);
        }
    }
    for row in rows {
        let own: Vec<&String> = row
            .notes
            .iter()
            .filter(|n| !shared.contains(&n.as_str()))
            .collect();
        if own.is_empty() {
            continue;
        }
        println!("\n{}", row.origin.install_target());
        for note in own {
            println!("  - {note}");
        }
    }
}

/// The counted column, which is blank rather than zero when it is unknown.
///
/// A zero here would sort and read as the CHEAPEST option for the model about
/// which the least is known, which is the inverse of its real rank -- the same
/// inversion an unsized ggml type once produced when it was printed as 0 bytes.
fn counted_column(fit: &catalog::Fit) -> String {
    match fit.counted_source {
        catalog::CountedSource::Unknown => "?".to_string(),
        catalog::CountedSource::Measured => format!("{}*", human_bytes(fit.counted)),
        catalog::CountedSource::Estimated => format!("~{}", human_bytes(fit.counted)),
    }
}

/// The verdict, plus the one flag that overrides everything above it.
///
/// `suspicious` is computed in the ranking and sinks a row to the bottom;
/// without printing it, a reader sees a 27B model at 600 MB sitting last for
/// no stated reason. `prism-ml/Bonsai-27B-gguf` is that row on this machine
/// today.
fn verdict_column(row: &catalog::Recommendation) -> String {
    if row.suspicious {
        format!(
            "{} -- far smaller than its name claims",
            row.fit.verdict.as_str()
        )
    } else {
        row.fit.verdict.as_str().to_string()
    }
}

fn tok_s_column(row: &catalog::Recommendation) -> String {
    match &row.measured {
        Some(m) => format!("{:.0}-{:.0}", m.decode_tok_s_min, m.decode_tok_s_max),
        None => "-".to_string(),
    }
}

fn truncate(text: &str, width: usize) -> String {
    if text.len() <= width {
        text.to_string()
    } else {
        format!("{}...", &text[..width.saturating_sub(3)])
    }
}
