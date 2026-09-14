import Foundation
import TurboSpark

/// Every decision the steering UI makes, as pure functions.
///
/// **PURE ON PURPOSE, and for `swift/CLAUDE.md` Gotcha 26's reason.** An
/// assertion written against `AppModel` needs a real install to reach, so it
/// never runs; a decision that lives here is testable with a fixture and a
/// couple of integers. `ReasoningLevelPolicy` and `ServerStatusRows` are the
/// same move for the same reason.
public enum AppSteeringPolicy {
    /// The behavior-affecting steering values captured for an open session.
    /// Display-only preset metadata is deliberately absent.
    public struct Configuration: Equatable, Sendable {
        public var active: Bool
        public var vectorPath: String?
        public var mode: AppSteeringModeOption?
        public var scale: Double?
        public var layers: String?
        public var target: Double?
        public var gate: Double?
    }

    /// Canonicalizes intent exactly as `buildOpenOptions` does.
    public static func configuration(
        enabled: Bool,
        preset: AppSteeringPreset?
    ) -> Configuration {
        let path = preset?.vectorPath.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard enabled, !path.isEmpty, let preset else {
            return Configuration(
                active: false, vectorPath: nil, mode: nil, scale: nil,
                layers: nil, target: nil, gate: nil)
        }
        let layers = preset.layers.trimmingCharacters(in: .whitespacesAndNewlines)
        return Configuration(
            active: true, vectorPath: path, mode: preset.mode, scale: preset.scale,
            layers: layers.isEmpty ? nil : layers,
            target: preset.mode == .clamp ? preset.target : nil,
            gate: preset.gate > 0 ? preset.gate : nil)
    }

    /// Whether a preset's vector can be loaded against a given install.
    ///
    /// **THIS MIRRORS `SteeringSet::validate` AND NOTHING MORE**, which is
    /// the honest scope: the engine refuses a width mismatch and a layer
    /// overrun, and refuses NOTHING else.
    public enum Compatibility: Equatable {
        /// The shapes line up. **NOT "compatible" -- see `summary`.**
        case shapeMatches
        case widthMismatch(vector: Int, model: Int)
        case layerOverrun(vectorSpans: Int, modelLayers: Int)
        /// Something needed for the check is not known: the file was never
        /// read, or the install declares no `hiddenSize`. Reported as unknown
        /// rather than assumed either way (`swift/CLAUDE.md` Gotcha 23).
        case unknown(String)
        /// No vector path at all. A preset in this state steers nothing.
        case noVector

        public var allowsEnabling: Bool {
            switch self {
            case .shapeMatches: return true
            // Deliberately PERMISSIVE. A width this app could not read is not
            // evidence of a mismatch, and refusing on it would make every
            // preset restored from an older settings file unusable until it
            // is re-registered. The engine still refuses at open, with the
            // numbers, which is a worse place to find out but not a silent
            // one.
            case .unknown: return true
            case .widthMismatch, .layerOverrun, .noVector: return false
            }
        }

        /// One line for a row, and it never says "compatible".
        ///
        /// **A SHAPE MATCH IS NOT A SEMANTIC MATCH.** `docs/OBLITERATION.md`
        /// states it plainly: a vector extracted for a different checkpoint
        /// of the same hidden size opens, steers, and changes behaviour in a
        /// direction nobody asked for, silently. The only thing this app can
        /// check is the shape, so the shape is what it claims.
        public var summary: String {
            switch self {
            case .shapeMatches:
                return "Shape matches this model. Whether the direction MEANS anything here is "
                    + "not checkable: a vector extracted for another checkpoint of the same "
                    + "width will load and steer something."
            case let .widthMismatch(vector, model):
                return "Vector is \(vector) wide, this model is \(model). It would be refused "
                    + "at load."
            case let .layerOverrun(spans, layers):
                return "Vector spans \(spans) blocks, this model has \(layers). It would be "
                    + "refused at load."
            case let .unknown(why):
                return "Shape not checked: \(why)"
            case .noVector:
                return "No control vector file set."
            }
        }
    }

    /// Compares a preset's recorded vector shape against an install's.
    ///
    /// `modelHidden` and `modelLayers` come from the install's own
    /// `manifest.json` (`arch.hiddenSize`, `arch.numLayers`), which is the
    /// only per-install measurement available with no session open.
    public static func compatibility(
        preset: AppSteeringPreset?,
        modelHidden: Int?,
        modelLayers: Int?
    ) -> Compatibility {
        guard let preset else { return .noVector }
        if preset.vectorPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .noVector
        }
        guard let vectorHidden = preset.vectorHidden else {
            return .unknown("this vector's header has not been read yet")
        }
        guard let modelHidden else {
            return .unknown("this model's manifest declares no hidden size")
        }
        if vectorHidden != modelHidden {
            return .widthMismatch(vector: vectorHidden, model: modelHidden)
        }
        // The layer check is SECOND and only runs once the widths agree,
        // matching `SteeringSet::validate`'s own order: a width mismatch is
        // the more fundamental fact and reporting the layer count beside it
        // would bury it.
        if let spans = preset.vectorSpannedLayers, let modelLayers, spans > modelLayers {
            return .layerOverrun(vectorSpans: spans, modelLayers: modelLayers)
        }
        return .shapeMatches
    }

    /// Why the steering control is disabled, or `nil` when it is usable.
    ///
    /// **DISABLED WITH A REASON, NEVER HIDDEN** (`swift/CLAUDE.md` Gotchas 23
    /// and 33): a missing row reads as a missing feature, a greyed one with a
    /// reason tells the user what to fix.
    ///
    /// `familySupported` is `info.steering.supported` and is `nil` before any
    /// session exists. Unknown is treated as usable -- the alternative is a
    /// control that is dead until a model happens to be loaded, which is the
    /// state a user reaches for it in.
    public static func disabledReason(
        familySupported: Bool?,
        familyReason: String?,
        preset: AppSteeringPreset?,
        compatibility: Compatibility
    ) -> String? {
        if familySupported == false {
            return familyReason
                ?? "This model's family does not dispatch the steering edit, so a control "
                    + "vector here would be refused at load."
        }
        guard preset != nil else {
            return "No steering preset selected. Add one in Settings > Safety and Steering."
        }
        if case .noVector = compatibility {
            return "This preset has no control vector file, so it would steer nothing."
        }
        if !compatibility.allowsEnabling {
            return compatibility.summary
        }
        return nil
    }

    /// Whether the session currently open is running what the settings ask
    /// for.
    ///
    /// **STEERING RESOLVES ONCE, AT OPEN.** Changing a preset or toggling the
    /// switch changes nothing about the model already loaded, and until this
    /// existed the app said nothing about that -- a knob that appears to do
    /// something and does not is the exact failure this whole feature is
    /// about (`docs/OBLITERATION.md`'s null control is the load-bearing
    /// check for the same reason).
    ///
    /// Compares INTENT against what `info.steering` reports, so it is right
    /// whichever way the two disagree: turning steering on without reloading,
    /// and turning it off without reloading, both answer `true`.
    public static func needsReload(
        wantEnabled: Bool,
        wantPreset: AppSteeringPreset?,
        sessionSteering: SessionInfo.Steering?,
        loadedConfiguration: Configuration? = nil
    ) -> Bool {
        guard let sessionSteering else { return false }
        if let loadedConfiguration {
            return configuration(enabled: wantEnabled, preset: wantPreset) != loadedConfiguration
        }
        let wantActive =
            wantEnabled
            && !(wantPreset?.vectorPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ?? true)
        if wantActive != sessionSteering.active { return true }
        guard wantActive, let wantPreset else { return false }
        if sessionSteering.mode != wantPreset.mode.rawValue { return true }
        // Compared with a tolerance rather than by equality: the scale makes
        // a round trip through JSON and an f32 in the engine, so an exact
        // Double comparison would report a pending reload forever.
        if let scale = sessionSteering.scale, abs(scale - wantPreset.scale) > 1e-6 { return true }
        return false
    }
}
