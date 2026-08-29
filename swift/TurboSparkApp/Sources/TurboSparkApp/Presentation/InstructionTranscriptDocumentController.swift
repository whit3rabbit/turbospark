import AppKit
import Foundation

/// A single message rendered in the instruction transcript document.
public struct InstructionTranscriptMessage: Equatable, Sendable {
    /// Author role of the transcript message.
    public enum Role: Equatable, Sendable {
        /// User prompt message.
        case user
        /// Assistant response message.
        case assistant
    }

    /// Role of the message author.
    public let role: Role
    /// Primary text content of the message.
    public let content: String
    /// Optional thinking or reasoning text preceding the assistant response.
    public let reasoning: String

    /// Creates a new transcript message.
    public init(
        role: Role,
        content: String,
        reasoning: String = ""
    ) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
    }
}

/// Coordinates and optimizes attributed string mutations for the chat transcript text view.
@MainActor
public final class InstructionTranscriptDocumentController {
    /// Visual mutation classification applied during synchronization.
    public enum Mutation: Equatable {
        /// No textual changes were made.
        case none
        /// The document was completely rebuilt.
        case rebuilt
        /// Incremental content was appended to the active turn.
        case appended
        /// The active turn was completed and finalized.
        case finalized
    }

    /// Result of synchronizing conversation state into attributed string storage.
    public struct UpdateResult {
        /// The type of mutation performed on the storage.
        public let mutation: Mutation
        /// Character range corresponding to the active assistant response.
        public let assistantRange: NSRange

        /// Creates an update result.
        public init(mutation: Mutation, assistantRange: NSRange) {
            self.mutation = mutation
            self.assistantRange = assistantRange
        }
    }

    /// Latest user prompt text in the transcript.
    public private(set) var prompt = ""
    /// Historical messages currently rendered.
    public private(set) var history: [InstructionTranscriptMessage] = []
    /// Current assistant response text.
    public private(set) var response = ""
    /// Current reasoning text.
    public private(set) var reasoning = ""
    /// Whether the active turn has finished generation.
    public private(set) var isFinalized = false
    /// Whether the prefill reading placeholder is actively displayed.
    public private(set) var showsPrefillPlaceholder = false
    /// Range of the active assistant output within the attributed string.
    public private(set) var assistantRange = NSRange(location: 0, length: 0)
    private var prefillPlaceholderRange: NSRange?
    private var prefillDotCount = 0

    private let renderer: ResponseMarkdownRenderer

    /// Initializes a document controller with an optional markdown renderer.
    public init(renderer: ResponseMarkdownRenderer = ResponseMarkdownRenderer()) {
        self.renderer = renderer
    }

    /// Clamps selected character ranges to valid bounds within the storage length.
    public static func clampedRanges(
        _ ranges: [NSRange],
        toLength length: Int
    ) -> [NSRange] {
        ranges.map { range in
            let location = min(max(range.location, 0), length)
            let available = max(0, length - location)
            return NSRange(location: location, length: min(max(range.length, 0), available))
        }
    }

    /// Decides whether the view should automatically scroll to bottom following a mutation.
    public static func shouldScrollToBottom(
        wasAtBottom: Bool,
        mutation: Mutation
    ) -> Bool {
        wasAtBottom || mutation == .finalized
    }

    /// Decides whether the prefill reading placeholder animation should run.
    public static func shouldRunPrefillAnimation(
        response: String,
        reasoning: String,
        isTerminal: Bool,
        requested: Bool
    ) -> Bool {
        requested && response.isEmpty && reasoning.isEmpty && !isTerminal
    }

    /// Synchronizes message history and live streamed text into the provided text storage.
    @discardableResult
    public func synchronize(
        storage: NSMutableAttributedString,
        history: [InstructionTranscriptMessage],
        response: String,
        reasoning: String = "",
        isTerminal: Bool,
        showsPrefillPlaceholder: Bool = false
    ) -> UpdateResult {
        let responseChanged = response != self.response || reasoning != self.reasoning
        let displaysPrefillPlaceholder = Self.shouldRunPrefillAnimation(
            response: response,
            reasoning: reasoning,
            isTerminal: isTerminal,
            requested: showsPrefillPlaceholder)
        let needsRebuild = history != self.history
            || !response.hasPrefix(self.response)
            || !reasoning.hasPrefix(self.reasoning)
            || (isFinalized && !isTerminal)
            || displaysPrefillPlaceholder != self.showsPrefillPlaceholder

        var mutation: Mutation = .none
        if needsRebuild
            || storage.length == 0
                && (!history.isEmpty || !response.isEmpty || !reasoning.isEmpty || displaysPrefillPlaceholder) {
            rebuild(
                storage: storage,
                history: history,
                response: response,
                reasoning: reasoning,
                showsPrefillPlaceholder: displaysPrefillPlaceholder)
            mutation = .rebuilt
        } else if response.count > self.response.count || reasoning.count > self.reasoning.count {
            rebuild(
                storage: storage,
                history: history,
                response: response,
                reasoning: reasoning,
                showsPrefillPlaceholder: displaysPrefillPlaceholder)
            mutation = .appended
        }

        self.history = history
        self.prompt = history.last(where: { $0.role == .user })?.content ?? ""
        self.response = response
        self.reasoning = reasoning
        self.showsPrefillPlaceholder = displaysPrefillPlaceholder

        if isTerminal && (!isFinalized || responseChanged) {
            rebuild(
                storage: storage,
                history: history,
                response: response,
                reasoning: reasoning,
                showsPrefillPlaceholder: false)
            isFinalized = true
            mutation = .finalized
        } else if !isTerminal {
            isFinalized = false
        }

        return UpdateResult(mutation: mutation, assistantRange: assistantRange)
    }

    /// Advances the prefill placeholder animation frame in the text storage.
    @discardableResult
    public func advancePrefillAnimation(
        storage: NSMutableAttributedString
    ) -> Bool {
        guard showsPrefillPlaceholder, var range = prefillPlaceholderRange else {
            return false
        }
        prefillDotCount = (prefillDotCount + 1) % 4
        let replacement = NSAttributedString(
            string: Self.prefillPlaceholder(dotCount: prefillDotCount),
            attributes: Self.prefillPlaceholderAttributes())
        storage.replaceCharacters(in: range, with: replacement)
        range.length = replacement.length
        prefillPlaceholderRange = range
        return true
    }

    private func rebuild(
        storage: NSMutableAttributedString,
        history: [InstructionTranscriptMessage],
        response: String,
        reasoning: String,
        showsPrefillPlaceholder: Bool
    ) {
        let document = NSMutableAttributedString()
        for message in history {
            switch message.role {
            case .user:
                document.append(NSAttributedString(
                    string: "You\n",
                    attributes: Self.userLabelAttributes()))
                document.append(NSAttributedString(
                    string: message.content,
                    attributes: Self.promptAttributes()))
            case .assistant:
                document.append(NSAttributedString(
                    string: "Assistant\n",
                    attributes: Self.assistantLabelAttributes()))
                if !message.reasoning.isEmpty {
                    document.append(NSAttributedString(
                        string: "Thinking:\n" + message.reasoning + "\n\n",
                        attributes: Self.reasoningAttributes()))
                }
                document.append(renderer.render(message.content).attributedString)
            }
            document.append(NSAttributedString(
                string: "\n\n",
                attributes: Self.promptAttributes()))
        }

        if !response.isEmpty || !reasoning.isEmpty || showsPrefillPlaceholder || !history.isEmpty {
            document.append(NSAttributedString(
                string: "Assistant\n",
                attributes: Self.assistantLabelAttributes()))
            assistantRange = NSRange(location: document.length, length: 0)

            if !reasoning.isEmpty {
                let reasoningAttr = NSAttributedString(
                    string: "Thinking:\n" + reasoning + "\n\n",
                    attributes: Self.reasoningAttributes())
                document.append(reasoningAttr)
            }

            if !response.isEmpty {
                let responseAttr = renderer.render(response).attributedString
                document.append(responseAttr)
                assistantRange.length = (document.length - assistantRange.location)
            }

            prefillDotCount = 0
            prefillPlaceholderRange = nil
            if showsPrefillPlaceholder && response.isEmpty && reasoning.isEmpty {
                let placeholder = NSAttributedString(
                    string: Self.prefillPlaceholder(dotCount: prefillDotCount),
                    attributes: Self.prefillPlaceholderAttributes())
                prefillPlaceholderRange = NSRange(
                    location: document.length,
                    length: placeholder.length)
                document.append(placeholder)
            }
        }

        storage.setAttributedString(document)
        isFinalized = false
    }

    private static func userLabelAttributes() -> [NSAttributedString.Key: Any] {
        labelAttributes(color: .secondaryLabelColor)
    }

    private static func assistantLabelAttributes() -> [NSAttributedString.Key: Any] {
        labelAttributes(color: TurboSparkTheme.accentNSColor)
    }

    private static func labelAttributes(
        color: NSColor
    ) -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.paragraphSpacing = 5
        return [
            .font: NSFont.systemFont(
                ofSize: NSFont.smallSystemFontSize,
                weight: .semibold),
            .foregroundColor: color,
            .paragraphStyle: style,
        ]
    }

    private static func promptAttributes() -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        return [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: style,
        ]
    }

    private static func reasoningAttributes() -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 2
        style.paragraphSpacing = 4
        style.firstLineHeadIndent = 8
        style.headIndent = 8
        return [
            .font: NSFont.monospacedSystemFont(ofSize: NSFont.systemFontSize - 1.5, weight: .regular),
            .foregroundColor: NSColor.secondaryLabelColor,
            .paragraphStyle: style,
        ]
    }

    private static func responseAttributes() -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 6
        return [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: style,
        ]
    }

    private static func prefillPlaceholderAttributes() -> [NSAttributedString.Key: Any] {
        var attributes = responseAttributes()
        attributes[.foregroundColor] = NSColor.secondaryLabelColor
        return attributes
    }

    private static func prefillPlaceholder(dotCount: Int) -> String {
        "Reading prompt" + String(repeating: ".", count: dotCount)
    }
}
