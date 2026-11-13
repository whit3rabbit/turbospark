use minijinja::ErrorKind;

/// Env override for [`strftime_now`]'s clock, as `YYYY-MM-DD` or a full
/// `YYYY-MM-DDTHH:MM:SS`.
///
/// **THIS EXISTS BECAUSE A TEMPLATE THAT READS THE CLOCK CANNOT BE FROZEN**,
/// and this repo's entire quality apparatus is frozen digests over a fixed
/// prompt. gpt-oss's Harmony template writes `Current date: ` into its system
/// preamble, so without an override its rendered prompt changes at midnight
/// and every digest taken over it expires the next day -- which reads as a
/// numerics regression rather than as a calendar.
///
/// The DEFAULT is the real clock, because that is what transformers, vLLM and
/// llama.cpp all send and the model was trained against a real date. Pinning
/// it unconditionally would make every user's prompt differ from the reference
/// implementations' to buy a convenience only the gates need.
pub const CHAT_DATE_ENV: &str = "TURBOSPARK_CHAT_DATE";

/// `strftime_now(format)`, the transformers chat-template global.
///
/// Implemented here rather than pulled in with a date crate: the only thing
/// needed is a broken-down UTC timestamp, and the civil-from-days conversion
/// is a dozen lines (Howard Hinnant's algorithm, valid for any proleptic
/// Gregorian date).
///
/// TWO DEVIATIONS FROM transformers, both deliberate and both stated rather
/// than silent. It is UTC where `datetime.now()` is LOCAL, which can differ by
/// a day at the boundary and reaches the model as one field of a system
/// preamble. And it supports the specifiers real chat templates actually use
/// (`%Y %m %d %H %M %S %y %e` and `%%`); an unrecognized one is emitted
/// VERBATIM, including its `%`, so an unsupported format is visible in the
/// prompt rather than silently dropped or guessed at.
pub(crate) fn strftime_now(format: String) -> Result<String, minijinja::Error> {
    let secs = match std::env::var(CHAT_DATE_ENV) {
        Ok(pinned) => parse_pinned(&pinned).ok_or_else(|| {
            minijinja::Error::new(
                ErrorKind::InvalidOperation,
                format!("{CHAT_DATE_ENV}={pinned:?} is not YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"),
            )
        })?,
        Err(_) => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0),
    };
    Ok(format_utc(secs, &format))
}

fn parse_pinned(value: &str) -> Option<i64> {
    let (date, time) = match value.split_once('T') {
        Some((d, t)) => (d, t),
        None => (value, "00:00:00"),
    };
    let mut d = date.split('-');
    let (y, m, day) = (
        d.next()?.parse::<i64>().ok()?,
        d.next()?.parse::<i64>().ok()?,
        d.next()?.parse::<i64>().ok()?,
    );
    // The day must fit the month (leap years included), or 2026-02-31 would
    // silently roll into March.
    if d.next().is_some() || !(1..=12).contains(&m) || !(1..=days_in_month(y, m)).contains(&day) {
        return None;
    }
    let mut t = time.split(':');
    let (hh, mm, ss) = (
        t.next()?.parse::<i64>().ok()?,
        t.next()?.parse::<i64>().ok()?,
        t.next().unwrap_or("0").parse::<i64>().ok()?,
    );
    // Python datetime's ranges (the transformers reference). Out-of-range
    // time fields would roll into another day, and an unbounded year
    // overflows i64 below (wrapping silently in release). With these bounds
    // the arithmetic cannot overflow.
    if t.next().is_some()
        || !(1..=9999).contains(&y)
        || !(0..=23).contains(&hh)
        || !(0..=59).contains(&mm)
        || !(0..=59).contains(&ss)
    {
        return None;
    }
    Some(days_from_civil(y, m, day) * 86_400 + hh * 3600 + mm * 60 + ss)
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Hinnant).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse, plus the time of day.
fn civil_from_secs(secs: i64) -> (i64, i64, i64, i64, i64, i64) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d, rem / 3600, (rem % 3600) / 60, rem % 60)
}

fn format_utc(secs: i64, format: &str) -> String {
    let (y, m, d, hh, mm, ss) = civil_from_secs(secs);
    let mut out = String::with_capacity(format.len() + 8);
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:2}")),
            Some('H') => out.push_str(&format!("{hh:02}")),
            Some('M') => out.push_str(&format!("{mm:02}")),
            Some('S') => out.push_str(&format!("{ss:02}")),
            Some('%') => out.push('%'),
            // VERBATIM, `%` included: an unsupported specifier should be
            // visible in the prompt rather than silently dropped.
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_clock_accepts_documented_forms() {
        assert_eq!(parse_pinned("1970-01-01"), Some(0));
        assert_eq!(parse_pinned("1970-01-02T01:02:03"), Some(86_400 + 3723));
        assert_eq!(
            parse_pinned("9999-12-31T23:59:59").map(|s| format_utc(s, "%Y-%m-%d %H:%M:%S")),
            Some("9999-12-31 23:59:59".to_string())
        );
    }

    // An out-of-range time field used to roll silently into another day, so
    // the "pinned" prompt carried a date nobody asked for.
    #[test]
    fn pinned_clock_rejects_out_of_range_time_fields() {
        for bad in [
            "2026-01-01T24:00:00",
            "2026-01-01T-1:00:00",
            "2026-01-01T00:60:00",
            "2026-01-01T00:00:60",
        ] {
            assert_eq!(parse_pinned(bad), None, "{bad}");
        }
    }

    // A huge year overflowed i64 in days_from_civil and the `* 86_400`:
    // a panic in debug, a silently wrapped date in release (no overflow checks).
    #[test]
    fn pinned_clock_rejects_years_that_overflow() {
        for bad in [
            "9223372036854775807-01-01",
            "-9223372036854775807-01-01",
            "10000-01-01",
            "2026-01-01T9223372036854775807:00:00",
        ] {
            assert_eq!(parse_pinned(bad), None, "{bad}");
        }
    }

    // The day was range-checked as 1..=31 only, so impossible calendar dates
    // rolled into the next month.
    #[test]
    fn pinned_clock_validates_day_against_month() {
        for bad in [
            "2026-02-31",
            "2026-02-29",
            "2026-04-31",
            "2026-06-31",
            "1900-02-29",
            "2026-02-00",
        ] {
            assert_eq!(parse_pinned(bad), None, "{bad}");
        }
        for ok in [
            "2024-02-29",
            "2000-02-29",
            "2026-02-28",
            "2026-04-30",
            "2026-12-31",
        ] {
            assert!(parse_pinned(ok).is_some(), "{ok}");
        }
    }
}
