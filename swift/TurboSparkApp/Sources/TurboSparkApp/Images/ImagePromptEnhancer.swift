import Foundation

/// Suggestion state for the Images composer's Enhance action.
///
/// The suggestion is only ever offered: Accept hands the text to the view to
/// put in the field, Dismiss drops it, and nothing here generates an image.
/// Any edit to the field that the suggestion was made from cancels the run or
/// drops the suggestion, so a stale rewrite can never replace newer typing.
@MainActor
final class ImagePromptEnhancer: ObservableObject {
    enum Phase: Equatable {
        case idle
        case running(source: String)
        case suggestion(source: String, text: String)
        /// The model answered with nothing usable.
        case failed(source: String)
    }

    @Published private(set) var phase: Phase = .idle
    /// The prompt Accept replaced, while the field still holds the accepted
    /// text. Any other edit ends the chance to undo, as with a text field.
    @Published private(set) var undoPrompt: String?
    private var acceptedText: String?

    private var task: Task<Void, Never>?
    /// Identifies the current run so a cancelled run's late result is dropped
    /// even if its completion ignored cancellation.
    private var runID = UUID()

    var isRunning: Bool {
        if case .running = phase { return true }
        return false
    }

    func start(prompt: String, completion: @escaping ImagePromptEnhancement.Completion) {
        cancelRun()
        clearUndo()
        let id = UUID()
        runID = id
        phase = .running(source: prompt)
        task = Task { [weak self] in
            let suggestion = await ImagePromptEnhancement.suggest(prompt: prompt, completion: completion)
            guard let self, !Task.isCancelled, self.runID == id else { return }
            self.phase = suggestion.map { .suggestion(source: prompt, text: $0) } ?? .failed(source: prompt)
        }
    }

    /// Returns the text to put in the field, or nil when there is no suggestion.
    func accept() -> String? {
        guard case .suggestion(let source, let text) = phase else { return nil }
        phase = .idle
        undoPrompt = source
        acceptedText = text
        return text
    }

    /// Returns the prompt to restore, or nil when Undo no longer applies.
    func undoAccept() -> String? {
        defer { clearUndo() }
        return undoPrompt
    }

    func dismiss() {
        cancelRun()
        phase = .idle
    }

    /// Called on every change of the prompt field.
    func promptChanged(to text: String) {
        switch phase {
        case .running(let source), .suggestion(let source, _), .failed(let source):
            if text != source { dismiss() }
        case .idle:
            break
        }
        if let acceptedText, text != acceptedText { clearUndo() }
    }

    /// Waits for the current run to finish. For tests.
    func waitForRun() async {
        await task?.value
    }

    private func cancelRun() {
        task?.cancel()
        runID = UUID()
    }

    private func clearUndo() {
        undoPrompt = nil
        acceptedText = nil
    }
}
