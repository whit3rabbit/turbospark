import Foundation

/// Utility helper providing human-readable relative and exact timestamp formatting for chat messages.
public enum MessageTimestampFormatter {
    private static let exactDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()

    private static let exactFullDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEEE, MMMM d, yyyy 'at' h:mm a"
        return formatter
    }()

    private static let relativeDateTimeFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter
    }()

    /// Returns a human-friendly relative string describing how long ago the date was.
    /// E.g. "Just now", "20 minutes ago", "2 hours ago", "Yesterday".
    public static func relativeString(for date: Date, relativeTo now: Date = Date()) -> String {
        let interval = now.timeIntervalSince(date)

        if interval < 0 {
            return "Just now"
        }

        if interval < 45 {
            return "Just now"
        }

        let calendar = Calendar.current
        if calendar.isDateInToday(date) {
            let minutes = max(1, Int(round(interval / 60.0)))
            if minutes < 60 {
                return minutes == 1 ? "1 minute ago" : "\(minutes) minutes ago"
            }
            let hours = max(1, Int(round(interval / 3600.0)))
            return hours == 1 ? "1 hour ago" : "\(hours) hours ago"
        }

        if calendar.isDateInYesterday(date) {
            let timeFormatter = DateFormatter()
            timeFormatter.timeStyle = .short
            return "Yesterday at \(timeFormatter.string(from: date))"
        }

        return relativeDateTimeFormatter.localizedString(for: date, relativeTo: now)
    }

    /// Returns the exact timestamp string.
    /// E.g. "Sunday, August 30, 2026 at 11:06 AM".
    public static func exactString(for date: Date) -> String {
        exactFullDateFormatter.string(from: date)
    }

    /// Returns standard medium date/time string.
    /// E.g. "Aug 30, 2026, 11:06 AM".
    public static func standardString(for date: Date) -> String {
        exactDateFormatter.string(from: date)
    }
}
