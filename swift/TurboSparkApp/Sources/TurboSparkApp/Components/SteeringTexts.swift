import SwiftUI

/// Localized renderings of the steering policy's English strings.
///
/// `AppSteeringPolicy` stays the authority on WHETHER a direction is usable
/// and builds its reasons as plain English (its tests assert on them). The
/// Safety pane and the editor sheet render those decisions from the catalog
/// instead, one literal per case so the catalog scan can see every key.
enum SteeringText {
    static func mode(_ mode: AppSteeringModeOption) -> Text {
        switch mode {
        case .ablate: return Text("Ablate (Suppress)", bundle: .module)
        case .add: return Text("Add (Amplify)", bundle: .module)
        case .clamp: return Text("Clamp", bundle: .module)
        case .renorm: return Text("Renorm", bundle: .module)
        }
    }

    /// Mirrors `AppSteeringPolicy.Compatibility.summary`. It never says
    /// "compatible": a shape match is not a semantic match.
    static func compatibility(_ compatibility: AppSteeringPolicy.Compatibility) -> Text {
        switch compatibility {
        case .shapeMatches:
            return Text("Shape matches this model. Whether the direction MEANS anything here is not checkable: a vector extracted for another checkpoint of the same width will load and steer something.", bundle: .module)
        case let .widthMismatch(vector, model):
            return Text("Vector is \(vector) wide, this model is \(model). It would be refused at load.", bundle: .module)
        case let .layerOverrun(spans, layers):
            return Text("Vector spans \(spans) blocks, this model has \(layers). It would be refused at load.", bundle: .module)
        case let .unknown(why):
            // The two reasons the policy produces today; anything newer still
            // renders, with its English reason inside the localized frame.
            switch why {
            case "this vector's header has not been read yet":
                return Text("Shape not checked: this vector's header has not been read yet.", bundle: .module)
            case "this model's manifest declares no hidden size":
                return Text("Shape not checked: this model's manifest declares no hidden size.", bundle: .module)
            default:
                return Text("Shape not checked: \(why)", bundle: .module)
            }
        case .noVector:
            return Text("No control vector file set.", bundle: .module)
        }
    }

    /// Mirrors `AppSteeringPolicy.disabledReason` branch for branch. The
    /// policy's own answer decides whether there is a reason at all; this
    /// only decides how it reads.
    @MainActor
    static func disabledReason(_ model: AppModel) -> Text? {
        guard model.steeringDisabledReason != nil else { return nil }
        if model.steeringFamilySupported == false {
            // The engine's own refusal text, when it sent one, is not ours
            // to translate.
            if let reason = model.info?.steering.reason {
                return Text(verbatim: reason)
            }
            return Text("This model's family does not dispatch the steering edit, so a control vector here would be refused at load.", bundle: .module)
        }
        guard model.resolvedSteeringPreset != nil else {
            return Text("No steering preset selected. Add one in Settings > Safety and Steering.", bundle: .module)
        }
        let compatibility = model.steeringCompatibility
        if case .noVector = compatibility {
            return Text("This preset has no control vector file, so it would steer nothing.", bundle: .module)
        }
        return Self.compatibility(compatibility)
    }
}
