import SwiftUI

/// Which Voice Cloning cards are showing "needs attention", and when each was
/// last pinged.
///
/// Nothing is flagged until the user tries to go on. A card that is merely
/// unfinished looks calm, because a page that opens covered in warnings
/// teaches people to ignore them. Pressing Generate with steps missing (or a
/// step in the strip, or the "steps left" button) calls `show`, which turns
/// the edge on for those steps and sends a ping. The edge then follows the
/// controller: it goes away the moment the step is done.
@MainActor
final class VoiceCloneAttention: ObservableObject {
    @Published private(set) var attempted = false
    @Published private(set) var pings: [VoiceCloneStep: Int] = [:]

    func show(_ steps: [VoiceCloneStep]) {
        attempted = true
        for step in steps { pings[step, default: 0] += 1 }
    }

    /// Forget the attempt, when a run starts or the model changes.
    func reset() {
        attempted = false
    }

    func isOn(_ step: VoiceCloneStep, missing: [VoiceCloneStep]) -> Bool {
        attempted && missing.contains(step)
    }

    func ping(_ step: VoiceCloneStep) -> Int {
        pings[step] ?? 0
    }

    /// The strip's statuses with every flagged step shown as needing
    /// attention instead of to do.
    func flagged(_ statuses: [VoiceCloneStepStatus], missing: [VoiceCloneStep]) -> [VoiceCloneStepStatus] {
        guard attempted else { return statuses }
        return statuses.map { missing.contains($0.step) ? VoiceCloneStepStatus(step: $0.step, state: .attention) : $0 }
    }

    /// The localized name of a step, for a VoiceOver announcement.
    static func name(of step: VoiceCloneStep) -> String {
        switch step {
        case .voice: return AppStrings.localized("Voice")
        case .consent: return AppStrings.localized("Consent")
        case .transcript: return AppStrings.localized("Transcript")
        case .script: return AppStrings.localized("Script")
        case .generate: return AppStrings.localized("Generate")
        }
    }
}
