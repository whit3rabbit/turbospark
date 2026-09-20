import Foundation

/// A short moving average smooths chunk callbacks. Uptime avoids wall-clock
/// changes, and querying between callbacks lets a stalled transfer fall to zero.
struct DownloadTransferMeter {
    private struct Sample {
        var time: TimeInterval
        var bytes: UInt64
    }

    private var samples: [Sample] = []
    private let window: TimeInterval = 10

    mutating func reset(bytes: UInt64, at time: TimeInterval) {
        samples = [Sample(time: time, bytes: bytes)]
    }

    mutating func record(bytes: UInt64, at time: TimeInterval) {
        guard let last = samples.last else {
            reset(bytes: bytes, at: time)
            return
        }
        guard time >= last.time, bytes >= last.bytes else { return }
        if time == last.time {
            samples[samples.count - 1].bytes = bytes
        } else {
            samples.append(Sample(time: time, bytes: bytes))
        }
        // Keep one sample before the window so its boundary can be interpolated.
        while samples.count > 2, samples[1].time <= time - window {
            samples.removeFirst()
        }
    }

    func bytesPerSecond(at time: TimeInterval) -> Double? {
        guard let first = samples.first, let last = samples.last,
              samples.count > 1, time >= last.time, time - first.time >= 1
        else { return nil }
        let start = max(first.time, time - window)
        guard start < last.time else { return 0 }
        var initialBytes = Double(first.bytes)
        for index in samples.indices.dropFirst() {
            let previous = samples[index - 1]
            let next = samples[index]
            if next.time >= start {
                let fraction = max(0, (start - previous.time) / (next.time - previous.time))
                initialBytes = Double(previous.bytes) + Double(next.bytes - previous.bytes) * fraction
                break
            }
        }
        return max(0, (Double(last.bytes) - initialBytes) / (time - start))
    }
}
