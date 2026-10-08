import Foundation

/// Utility helper providing human-readable relative and exact timestamp formatting for chat messages.
public enum MessageTimestampFormatter {
    /// Formatters are built per call: they must follow the in-app language
    /// (passed in from the view's `\.locale`), not the system locale, and
    /// these run for a handful of visible rows at a time.
    private static func relativeFormatter(_ locale: Locale) -> RelativeDateTimeFormatter {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        formatter.dateTimeStyle = .named
        formatter.locale = locale
        return formatter
    }

    /// Returns a human-friendly relative string describing how long ago the date was.
    /// E.g. "Just now", "20 minutes ago", "2 hours ago", "yesterday".
    public static func relativeString(
        for date: Date, relativeTo now: Date = Date(), locale: Locale = .current
    ) -> String {
        let interval = now.timeIntervalSince(date)

        if interval < 45 {
            return String(localized: "Just now", bundle: .module, locale: locale)
        }

        let formatter = relativeFormatter(locale)
        // Keep short elapsed times relative even when they cross local midnight.
        if interval < 24 * 60 * 60 {
            let minutes = max(1, Int(round(interval / 60.0)))
            if minutes < 60 {
                return formatter.localizedString(from: DateComponents(minute: -minutes))
            }
            let hours = max(1, Int(round(interval / 3600.0)))
            return formatter.localizedString(from: DateComponents(hour: -hours))
        }

        return formatter.localizedString(for: date, relativeTo: now)
    }

    /// Returns the exact timestamp string in the given locale's own pattern.
    /// E.g. "Sunday, August 30, 2026 at 11:06 AM" for en_US.
    public static func exactString(for date: Date, locale: Locale = .current) -> String {
        let formatter = DateFormatter()
        formatter.locale = locale
        formatter.setLocalizedDateFormatFromTemplate("EEEEMMMMdyjmm")
        return formatter.string(from: date)
    }

    /// Returns standard medium date/time string.
    /// E.g. "Aug 30, 2026, 11:06 AM".
    public static func standardString(for date: Date, locale: Locale = .current) -> String {
        let formatter = DateFormatter()
        formatter.locale = locale
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    /// Returns a compact relative time representation such as "now", "5m", "2h", "52d".
    public static func compactRelativeString(for date: Date, relativeTo now: Date = Date()) -> String {
        let interval = now.timeIntervalSince(date)
        if interval < 60 {
            return "now"
        }
        let minutes = Int(interval / 60.0)
        if minutes < 60 {
            return "\(minutes)m"
        }
        let hours = Int(interval / 3600.0)
        if hours < 24 {
            return "\(hours)h"
        }
        let days = Int(interval / 86400.0)
        return "\(days)d"
    }
}
