import Foundation

/// State for the composer's shared `/` and `@` autocomplete popup.
///
/// Owns WHICH suggestion is highlighted and WHAT the popup is showing; the
/// matching itself lives in `ComposerAutocompleteEngine` (pure, testable)
/// and the file listing in `ProjectFileIndex`. The controller never writes
/// the composer text: `accept(in:)` RETURNS the replacement text and the
/// view applies it, so accepting stays testable without a view.
@MainActor
final class ComposerAutocompleteController: ObservableObject {
    /// Suggestions currently offered, best first.
    @Published private(set) var suggestions: [ComposerSuggestion] = []
    /// Set when the popup is visible with nothing (yet) to choose: a scan
    /// in flight, or the no-project hint.
    @Published private(set) var hint: String?
    /// Highlighted row, an index into `suggestions`.
    @Published private(set) var selectedIndex: Int = 0

    private(set) var triggerMatch: ComposerTriggerMatch?
    /// The trigger Escape dismissed. Stays closed while that exact trigger
    /// is unchanged and reopens the moment the token grows or moves.
    private var dismissedMatch: ComposerTriggerMatch?
    /// Bumps on every state reset so an in-flight file scan for a trigger
    /// that has since gone stale applies to nothing.
    private var scanGeneration = 0
    private var lastSkills: [AppSkill] = []
    private var lastProjectRoot: URL?

    var isVisible: Bool {
        triggerMatch != nil && dismissedMatch != triggerMatch
            && (!suggestions.isEmpty || hint != nil)
    }

    /// Recomputes the popup from a text change. Call on every keystroke.
    func textChanged(text: String, skills: [AppSkill], projectRoot: URL?) {
        lastSkills = skills
        lastProjectRoot = projectRoot
        guard let match = ComposerAutocompleteEngine.detectTrigger(in: text) else {
            reset()
            return
        }
        if dismissedMatch == match {
            // Still editing the exact token that was dismissed: stay closed
            // without tearing the match down, so Escape-then-more-typing
            // does not flash the popup between keystrokes.
            return
        }
        dismissedMatch = nil
        triggerMatch = match
        recompute()
    }

    func focusChanged(isFocused: Bool) {
        guard !isFocused else { return }
        reset()
    }

    /// Escape: close the popup for this trigger only.
    func dismiss() {
        dismissedMatch = triggerMatch
        objectWillChange.send()
    }

    /// Moves the highlight, wrapping at both ends.
    func moveSelection(_ delta: Int) {
        guard !suggestions.isEmpty else { return }
        let count = suggestions.count
        selectedIndex = ((selectedIndex + delta) % count + count) % count
    }

    /// Points the highlight at the row the user CLICKED, so the accept that
    /// follows (`accept(in:)` reads `selectedIndex`) inserts the clicked
    /// row rather than whichever row the keyboard happened to highlight.
    func select(_ index: Int) {
        guard suggestions.indices.contains(index) else { return }
        selectedIndex = index
        objectWillChange.send()
    }

    /// The full composer text with the trigger token replaced by the
    /// highlighted suggestion, or nil when there is nothing to accept. The
    /// trailing space is part of the result: a completed token is not a
    /// trigger, so the popup closes by the same text change that inserted
    /// it. A reference PREFIX row (`@file:`, `@git:`, `@url:`, `@folder:`)
    /// continues instead: no trailing space, and the token still ends in a
    /// live trigger, so the same text change re-opens the popup for what
    /// follows the colon. A COMPLETED row under the scheme
    /// (`@file:src/a.swift`) is an ordinary file row and takes the space.
    func accept(in text: String) -> String? {
        guard isVisible, let match = triggerMatch,
            suggestions.indices.contains(selectedIndex)
        else { return nil }
        let head = String(text.prefix(match.tokenStartOffset))
        let row = suggestions[selectedIndex]
        let continues = row.kind == .reference && row.replacementToken.hasSuffix(":")
        return head + row.replacementToken + (continues ? "" : " ")
    }

    // MARK: - Internals

    private func recompute() {
        guard let match = triggerMatch else { return }
        switch match.trigger {
        case .slash(let prefix):
            hint = nil
            setSuggestions(
                ComposerAutocompleteEngine.slashCandidates(prefix: prefix, skills: lastSkills))
        case .mention(let query):
            recomputeMention(query: query, projectRoot: lastProjectRoot)
        }
    }

    private func recomputeMention(query: String, projectRoot: URL?) {
        guard let projectRoot else {
            // The hint must be set BEFORE the (empty) suggestions, or the
            // empty-reset in `setSuggestions` tears the popup down before it
            // ever showed.
            hint = "Open a project to reference its files with @."
            setSuggestions([])
            return
        }
        // Show whatever is already known immediately (a stale snapshot beats
        // a flash of empty), plus the scanning hint while a rebuild runs.
        let known = ProjectFileIndex.shared.cachedEntries(for: projectRoot)
        hint = known == nil ? "Scanning project files..." : nil
        setSuggestions(ComposerAutocompleteEngine.mentionCandidates(query: query, entries: known ?? []))

        let generation = scanGeneration
        let match = triggerMatch
        Task { [weak self] in
            let entries = await ProjectFileIndex.shared.entries(for: projectRoot)
            guard let self else { return }
            guard self.scanGeneration == generation, self.triggerMatch == match else { return }
            self.setSuggestions(
                ComposerAutocompleteEngine.mentionCandidates(query: query, entries: entries))
            self.hint = nil
        }
    }

    private func setSuggestions(_ updated: [ComposerSuggestion]) {
        suggestions = updated
        selectedIndex = min(selectedIndex, max(updated.count - 1, 0))
        if updated.isEmpty && hint == nil {
            // Nothing matches and nothing to say: the popup hides rather
            // than showing an empty box (this is what keeps typing a plain
            // `/usr/bin` path or an email from flashing a popup).
            reset()
        }
    }

    private func reset() {
        scanGeneration += 1
        triggerMatch = nil
        dismissedMatch = nil
        suggestions = []
        hint = nil
        selectedIndex = 0
    }
}
