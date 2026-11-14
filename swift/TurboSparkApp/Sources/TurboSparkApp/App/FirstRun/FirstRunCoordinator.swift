import Foundation
import SwiftUI

// OC-07: first-run onboarding. This file is the model and the persistence;
// the views are `FirstRunFlowView` and `SplashOverlayView`. Nothing in the
// FirstRun folder may request a macOS privacy grant, a credential, or a
// network resource (`FirstRunNoPermissionPromptTests` scans the folder).

/// The one persisted fact: the welcome flow was finished or skipped.
struct FirstRunRecord: Codable, Equatable {
    var completed: Bool
    var completedAt: Date?
}

/// The completion flag, at APP level rather than inside a profile.
///
/// It lives under `AppStorageRoot.machineRoot` (shared by every profile) so a
/// second profile on the same Mac does not replay a welcome the person has
/// already seen. It goes through `AppJSONStore`: a corrupt file is
/// quarantined, never overwritten, and reads back as "no flag". The existing
/// user rule in `FirstRunDecision` is what keeps that case from replaying the
/// flow for someone who has history.
struct FirstRunStore {
    static let storeLabel: String.LocalizationValue = "Welcome setup"

    let fileURL: URL

    init(fileURL: URL = AppStorageRoot.machineRoot.appendingPathComponent("first_run.json")) {
        self.fileURL = fileURL
    }

    private static var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    private static var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }

    /// Nil when there is no file (a first run) or it was unreadable and has
    /// been quarantined.
    func load() -> FirstRunRecord? {
        AppJSONStore.load(
            FirstRunRecord.self, from: fileURL, label: Self.storeLabel, decoder: Self.decoder)
    }

    @discardableResult
    func markCompleted(now: Date = Date()) -> Bool {
        AppJSONStore.save(
            FirstRunRecord(completed: true, completedAt: now), to: fileURL,
            label: Self.storeLabel, encoder: Self.encoder)
    }
}

/// Whether this launch is a genuinely new install.
enum FirstRunDecision {
    /// Upgrading users must never be pushed through a welcome flow, and a
    /// flag that was lost or quarantined must not replay it either, so the
    /// flag alone is not enough: any chat with a message, or any installed
    /// model, means the person has used this app (or the shared model
    /// library) before and the flow does not run.
    static func shouldRun(
        flagCompleted: Bool, hasInstalledModel: Bool, hasChatHistory: Bool
    ) -> Bool {
        !flagCompleted && !hasInstalledModel && !hasChatHistory
    }
}

enum FirstRunStep: String, Equatable, CaseIterable {
    case welcome
    case firstModel
    /// Hands off to the desktop-shell-hardening import wizard. That wizard is
    /// not in the repository, so `FirstRunFlow.importWizardAvailable` is
    /// false and this step is ABSENT from every list, not a placeholder.
    case importHandoff
    case primer
}

enum FirstRunFlow {
    /// Flip only when the import wizard ships and `FirstRunFlowView` has a
    /// real view for `.importHandoff`.
    static let importWizardAvailable = false

    /// The ordered steps for this machine. A value, so the absent-step rules
    /// are tested without a view.
    static func steps(
        hasInstalledModel: Bool,
        importWizardAvailable: Bool = FirstRunFlow.importWizardAvailable
    ) -> [FirstRunStep] {
        var steps: [FirstRunStep] = [.welcome]
        if !hasInstalledModel { steps.append(.firstModel) }
        if importWizardAvailable { steps.append(.importHandoff) }
        steps.append(.primer)
        return steps
    }
}

/// Where the app should land when the flow ends.
enum FirstRunExit: Equatable {
    /// Finished or skipped: a new chat with the composer focused.
    case newChat
    /// The person chose Browse Models: the existing Discover section.
    case modelHub
}

/// The flow's state machine.
///
/// The flag is written exactly when the flow ENDS (finish, skip all, or
/// Browse Models). Quitting mid-flow writes nothing, so a relaunch shows the
/// flow again unless the person has meanwhile gained history or a model.
@MainActor
final class FirstRunCoordinator: ObservableObject {
    @Published private(set) var isActive = false
    @Published private(set) var steps: [FirstRunStep] = []
    @Published private(set) var index = 0

    /// Told once, when the flow ends, where to land. The view layer sets it;
    /// the coordinator takes plain values only.
    var onExit: ((FirstRunExit) -> Void)?

    private let store: FirstRunStore
    private var hasBegun = false

    init(store: FirstRunStore = FirstRunStore()) {
        self.store = store
    }

    var current: FirstRunStep? {
        steps.indices.contains(index) ? steps[index] : nil
    }
    var canGoBack: Bool { index > 0 }
    var isLastStep: Bool { index == steps.count - 1 }

    /// Decides once per process. A vault lock and unlock remounts the root
    /// view; the guard keeps that from restarting a flow that ended.
    func begin(hasInstalledModel: Bool, hasChatHistory: Bool) {
        guard !hasBegun else { return }
        hasBegun = true
        let flagCompleted = store.load()?.completed == true
        guard FirstRunDecision.shouldRun(
            flagCompleted: flagCompleted, hasInstalledModel: hasInstalledModel,
            hasChatHistory: hasChatHistory)
        else { return }
        steps = FirstRunFlow.steps(hasInstalledModel: hasInstalledModel)
        index = 0
        isActive = true
    }

    /// A model that finishes installing while the flow is open removes the
    /// first-model step, keeping the person on the step they are reading.
    func installedModelChanged(_ hasInstalledModel: Bool) {
        guard isActive else { return }
        let kept = current
        let updated = FirstRunFlow.steps(hasInstalledModel: hasInstalledModel)
        guard updated != steps else { return }
        steps = updated
        if let kept, let position = updated.firstIndex(of: kept) {
            index = position
        } else {
            // The step on screen is the one that went away: move to the next
            // step that survived, never past the end.
            index = min(index, updated.count - 1)
        }
    }

    /// Continue, or Get Started on the last step.
    func advance() {
        guard isActive else { return }
        if isLastStep {
            end(.newChat)
        } else {
            index += 1
        }
    }

    /// Skip this step. Same movement as Continue: no step has state to undo.
    func skipStep() { advance() }

    func back() {
        guard isActive, canGoBack else { return }
        index -= 1
    }

    func skipAll() { end(.newChat) }

    /// Browse Models: the person is acting on the first-model step, so the
    /// flow is done and does not replay. The primer is not shown on this
    /// path; every Allow sheet states its own reason at the point of asking.
    func browseModels() { end(.modelHub) }

    private func end(_ exit: FirstRunExit) {
        guard isActive else { return }
        store.markCompleted()
        isActive = false
        onExit?(exit)
    }
}
