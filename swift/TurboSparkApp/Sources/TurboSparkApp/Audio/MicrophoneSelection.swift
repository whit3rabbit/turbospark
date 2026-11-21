import Foundation

/// A saved microphone choice. It is kept by Core Audio device UID, because the
/// numeric AudioObjectID is reassigned across reboots and replugs and would
/// silently point at another device or at nothing.
enum MicrophoneSelection {
    struct Device: Identifiable, Equatable, Sendable {
        /// The current AudioObjectID, valid for this boot only.
        var id: UInt32
        /// The stable Core Audio UID that settings store.
        var uid: String
        var name: String
    }

    struct Resolution: Equatable {
        /// The device to open, or nil for the system default input.
        var deviceID: UInt32?
        /// A device was saved but is not connected now, so the default is
        /// used. The UI says so; the saved UID is kept for when it returns.
        var savedDeviceMissing: Bool
    }

    /// Never substitutes a different device for a missing one: the only
    /// fallback is the system default.
    static func resolve(savedUID: String?, available: [Device]) -> Resolution {
        guard let savedUID, !savedUID.isEmpty else { return Resolution(deviceID: nil, savedDeviceMissing: false) }
        guard let device = available.first(where: { $0.uid == savedUID }) else {
            return Resolution(deviceID: nil, savedDeviceMissing: true)
        }
        return Resolution(deviceID: device.id, savedDeviceMissing: false)
    }
}

/// Input level math for the microphone test meter.
enum MicrophoneLevel {
    /// The meter's bottom. Room noise on a laptop mic sits around -55 to
    /// -45 dBFS, so -60 keeps a quiet room visibly near empty.
    static let floorDecibels: Float = -60
    /// Speech at a normal distance reads about -35 to -20 dBFS. Below this
    /// the meter says it is waiting for you rather than that it hears you.
    static let hearingDecibels: Float = -42

    /// Root mean square of interleaved samples. A non-finite sample means a
    /// bad buffer, which reads as silence rather than a spike.
    static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for sample in samples {
            guard sample.isFinite else { return 0 }
            sum += sample * sample
        }
        return (sum / Float(samples.count)).squareRoot()
    }

    /// Maps an RMS level onto 0...1 linearly in decibels, which matches how
    /// loudness is heard; a linear RMS bar would sit near empty for speech.
    static func fraction(rms: Float) -> Double {
        guard rms.isFinite, rms > 0 else { return 0 }
        let decibels = 20 * log10(rms)
        let value = (decibels - floorDecibels) / -floorDecibels
        return Double(min(1, max(0, value)))
    }

    static func hearsInput(fraction: Double) -> Bool {
        fraction >= Double((hearingDecibels - floorDecibels) / -floorDecibels)
    }
}
