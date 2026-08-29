use super::*;
use crate::Catalog;

const GIB: u64 = 1024 * 1024 * 1024;

fn m4_max() -> Machine {
    Machine {
        physical_bytes: 36 * GIB,
        working_set_bytes: Some(27 * GIB),
        load_guard: model_io::LoadGuard::default(),
        chip: "Apple M4 Max".to_string(),
    }
}

fn ranked(machine: &Machine, context: u32) -> Vec<Recommendation> {
    let catalog = Catalog::embedded().expect("the embedded catalog parses");
    let entries: Vec<&CatalogEntry> = catalog.entries().collect();
    recommend_catalog(&entries, machine, context)
}

/// The development machine at the protocol window: the rows with frozen
/// evidence come first, and `gemma4` -- the install every frozen Gemma
/// row is measured against -- is one of them.
#[test]
fn the_measured_rows_lead_on_the_machine_they_were_measured_on() {
    let out = ranked(&m4_max(), 4096);
    let leaders: Vec<&str> = out
        .iter()
        .take(4)
        .map(|r| match &r.origin {
            Origin::Catalog(a) => a.as_str(),
            _ => unreachable!("no discovery in this arm"),
        })
        .collect();
    assert!(
        leaders.contains(&"gemma4"),
        "gemma4 should lead on this machine, got {leaders:?}"
    );
    for r in out.iter().take(4) {
        assert_eq!(r.evidence, Evidence::Verified);
        assert_eq!(r.fit.counted_source, CountedSource::Measured);
        // And every one of them says what configuration its number holds
        // at, because nothing offline knows what `Auto` would resolve to.
        assert!(
            r.notes.iter().any(|n| n.contains("expert-cache slots")),
            "{:?} quotes a measured peak with no configuration: {:?}",
            r.origin,
            r.notes
        );
    }
}

/// **The unprobed arm must not pass off the slot floor as a resolved
/// count.** With no header there is no expert stride, so `Auto` divides by
/// nothing and returns `DEFAULT_CACHE_SLOTS` -- which is 16, which is what
/// the protocol pins, which would make the two look like agreement. On
/// this machine `Auto` really resolves Gemma 4 to 32 once the stride is
/// known, so the offline number is a measurement at a stated
/// configuration and never a prediction about this one.
#[test]
fn an_unprobed_row_does_not_pass_off_the_slot_floor_as_a_resolved_count() {
    let catalog = Catalog::embedded().unwrap();
    let gemma = catalog.get("gemma4").unwrap();
    let r = from_entry(gemma, &m4_max(), 4096, None);
    assert_eq!(r.fit.counted_source, CountedSource::Measured);
    assert_eq!(r.fit.slots, 16, "the measured row's own configuration");
    assert!(r
        .notes
        .iter()
        .any(|n| n.contains("nothing here has read this checkpoint's header")));

    // **And the guard has to be able to SEE that**, which the row above
    // cannot show: the slot floor is 16 and the protocol pins 16, so
    // requiring them to match passes for the wrong reason. A row measured
    // at 32 slots is the input where the two rules part company -- it must
    // still apply offline, at its own stated configuration, rather than
    // being discarded for disagreeing with a count nobody resolved.
    let mut at_32 = gemma.clone();
    at_32.measured[0].expert_cache_slots = 32;
    at_32.measured[0].peak_footprint_mib = 3654;
    let r = from_entry(&at_32, &m4_max(), 4096, None);
    assert_eq!(r.fit.counted_source, CountedSource::Measured);
    assert_eq!(r.fit.slots, 32);
    assert_eq!(r.fit.counted, 3654 * 1024 * 1024);
}

/// **A measured peak at 4,096 is not applied to an 8,192 request**, and
/// the row says why rather than silently reporting the wrong number.
#[test]
fn a_measured_peak_does_not_travel_across_context_windows() {
    let out = ranked(&m4_max(), 8192);
    let gemma = out
        .iter()
        .find(|r| r.origin == Origin::Catalog("gemma4".into()))
        .expect("gemma4 is in the table");
    assert!(gemma.measured.is_some(), "the row is still reported");
    assert_ne!(gemma.fit.counted_source, CountedSource::Measured);
    assert!(
        gemma.notes.iter().any(|n| n.contains("4096 context")),
        "the mismatch has to be stated: {:?}",
        gemma.notes
    );
}

/// An unknown chip matches no measured row, which is correct: a peak
/// taken on an M4 Max says nothing about an M2. Every row falls back to
/// `Unknown` rather than borrowing another machine's number.
#[test]
fn an_unrecognized_chip_borrows_nobody_elses_measurements() {
    let machine = Machine {
        physical_bytes: 16 * GIB,
        working_set_bytes: None,
        load_guard: model_io::LoadGuard::default(),
        chip: "Some Other Silicon".to_string(),
    };
    for r in ranked(&machine, 4096) {
        assert!(
            r.measured.is_none(),
            "{:?} matched a foreign chip",
            r.origin
        );
        assert_eq!(r.fit.counted_source, CountedSource::Unknown);
    }
}

/// The working-set advisory is about the CANDIDATE, not the machine.
///
/// This machine's Metal working set (27 GiB) sits below its budget
/// (36 - 4 = 32 GiB), which is true of every unified-memory Mac over
/// 16 GB and says nothing about any particular model. What is worth
/// reporting is a candidate whose own allocations cross that line.
#[test]
fn the_working_set_advisory_is_about_the_candidate_and_not_the_machine() {
    let machine = m4_max();
    assert!(
        !machine.exceeds_working_set(3 * GIB),
        "an ordinary install is nowhere near the device limit"
    );
    assert!(machine.exceeds_working_set(30 * GIB));
    // No probe means no opinion, never a warning.
    assert!(!Machine {
        working_set_bytes: None,
        ..machine
    }
    .exceeds_working_set(300 * GIB));
}
