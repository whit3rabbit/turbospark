import Foundation

/// Formatting utilities for hardware metrics, timing durations, generation rates, and byte sizes.
enum MetricFormat {
    private static let memoryFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter
    }()

    private static let storageFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useMB, .useGB, .useTB]
        formatter.includesUnit = true
        formatter.isAdaptive = true
        return formatter
    }()

    /// Formats a time duration in seconds, displaying milliseconds for sub-second durations.
    static func seconds(_ value: Double?) -> String {
        guard let value else { return "\u{2014}" }
        if value < 1 { return String(format: "%.0f ms", value * 1000) }
        return String(format: "%.2f s", value)
    }

    /// Formats a time duration in milliseconds with one decimal place.
    static func milliseconds(_ value: Double?) -> String {
        guard let value else { return "\u{2014}" }
        return "\(value.formatted(.number.precision(.fractionLength(1)))) ms"
    }

    /// Formats a token throughput rate with one decimal place.
    static func rate(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    /// Formats a percentage value with one decimal place.
    static func percent(_ value: Double) -> String {
        String(format: "%.1f%%", value)
    }

    /// Formats memory byte count into a human-readable memory string.
    static func memory(_ bytes: UInt64?) -> String {
        guard let bytes else { return "\u{2014}" }
        return memoryFormatter.string(fromByteCount: Int64(bytes))
    }

    /// Formats disk storage byte count into a human-readable storage string.
    static func storage(_ bytes: UInt64) -> String {
        storageFormatter.string(fromByteCount: Int64(clamping: bytes))
    }
}
