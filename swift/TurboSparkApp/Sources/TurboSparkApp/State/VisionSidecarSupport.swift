import Foundation
import TurboSpark

/// Image input from a vision sidecar: a tower installed beside a model that
/// has none of its own (`turbospark-model pull-vision`, or a catalog row of
/// kind `vision-tower`).
enum VisionSidecar {
    /// The engine's "pair one by itself" spelling of `OpenOptions.visionSidecar`.
    /// It matches an installed tower to the model's family and width, does
    /// nothing for a model with its own tower, and leaves a model with no
    /// matching tower text-only.
    static let automatic = "auto"

    /// A sidecar is listed beside the models but is not one: it cannot be loaded.
    static func isAddOn(_ model: InstalledModel) -> Bool { model.kind == "vision-tower" }

    /// Whether an open failed on the sidecar step. The engine prefixes every
    /// such error with the option's name.
    static func isAttachFailure(_ message: String) -> Bool { message.contains("visionSidecar") }

    /// Opens with `options`, and once more without a sidecar when the sidecar
    /// is what failed: two candidate towers, or a tower whose image markers
    /// this model's tokenizer lacks. An optional add-on must never stop a
    /// model that loads without it. `skipped` is the engine's reason.
    @MainActor
    static func open<Session>(
        _ options: OpenOptions,
        with open: (OpenOptions) async throws -> Session
    ) async throws -> (session: Session, skipped: String?) {
        do {
            return (try await open(options), nil)
        } catch where options.visionSidecar != nil && isAttachFailure(error.localizedDescription) {
            var textOnly = options
            textOnly.visionSidecar = nil
            return (try await open(textOnly), error.localizedDescription)
        }
    }
}

extension AppModel {
    /// Says why a model loaded without the image input a sidecar offered.
    func announceVisionSidecarSkipped(alias: String, reason: String) {
        showToast(
            AppStrings.localized("Loaded \(alias) without image input. \(reason)"),
            style: .warning, duration: 8.0)
    }

    /// Answers a request to load a sidecar as if it were a model.
    func announceVisionSidecarIsNotAModel(_ model: InstalledModel) {
        showToast(
            AppStrings.localized(
                "\(model.alias) is a vision add-on. It attaches by itself when you load a matching model."),
            style: .info, duration: 6.0)
    }
}
