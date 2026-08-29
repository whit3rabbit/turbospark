import Foundation

/// Point-in-time progress snapshot of a model download stream.
public struct DownloadETAObservation: Sendable, Equatable {
    /// Total bytes downloaded so far.
    public let downloadedBytes: UInt64
    /// Total expected byte size of the target model artifact.
    public let totalBytes: UInt64

    public init(downloadedBytes: UInt64, totalBytes: UInt64) {
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
    }
}

/// Rolling exponential rate estimator for computing download transfer speeds and remaining time.
public struct DownloadETAEstimator {
    private var lastBytes: UInt64 = 0
    private var lastTime: Date?
    private var rollingRate: Double = 0

    public init() {}

    /// Resets the estimator state between download attempts.
    public mutating func reset() {
        lastBytes = 0
        lastTime = nil
        rollingRate = 0
    }

    /// Feeds a new progress sample and returns formatted ETA and speed string if available.
    ///
    /// - Parameters:
    ///   - downloaded: Bytes downloaded so far.
    ///   - total: Total expected size in bytes.
    /// - Returns: Human-readable speed and time remaining (e.g. "45.2 MB/s | 1m 20s left").
    public mutating func update(downloaded: UInt64, total: UInt64) -> String? {
        let now = Date()
        guard let lastTime = self.lastTime else {
            self.lastTime = now
            self.lastBytes = downloaded
            return nil
        }

        let elapsed = now.timeIntervalSince(lastTime)
        guard elapsed >= 0.5 else { return nil }

        let bytesDelta = downloaded > lastBytes ? Double(downloaded - lastBytes) : 0
        let instantRate = bytesDelta / elapsed
        if rollingRate == 0 {
            rollingRate = instantRate
        } else {
            rollingRate = 0.8 * rollingRate + 0.2 * instantRate
        }

        self.lastTime = now
        self.lastBytes = downloaded

        guard rollingRate > 1024, total > downloaded else { return nil }
        let remainingBytes = Double(total - downloaded)
        let seconds = remainingBytes / rollingRate

        return formatETA(seconds: seconds, bytesPerSec: rollingRate)
    }

    private func formatETA(seconds: Double, bytesPerSec: Double) -> String {
        let speed = MetricFormat.storage(UInt64(bytesPerSec)) + "/s"
        if seconds < 60 {
            return "\(speed) | \(Int(seconds))s left"
        }
        let minutes = Int(seconds / 60)
        let remSeconds = Int(seconds.truncatingRemainder(dividingBy: 60))
        if minutes < 60 {
            return "\(speed) | \(minutes)m \(remSeconds)s left"
        }
        let hours = Int(minutes / 60)
        let remMinutes = minutes % 60
        return "\(speed) | \(hours)h \(remMinutes)m left"
    }
}
