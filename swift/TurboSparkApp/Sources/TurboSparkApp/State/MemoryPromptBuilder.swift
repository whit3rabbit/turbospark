import Foundation

/// The transcript shape of a `#` quick-save, and its submit-time predicate.
///
/// Claude Code wraps the saved text in a user message tagged
/// `<user-memory-input>` and renders THAT as a memory badge; the message
/// then rides the transcript like any other turn content. This port keeps
/// the same shape so the saved note is visible to later turns, and the
/// renderer (`MessageRowView`) uses `parse` to show it as a memory row
/// instead of raw tags.
public enum UserMemoryInputMessage {
    public static let openTag = "<user-memory-input>"
    public static let closeTag = "</user-memory-input>"

    /// Whether a trimmed composer draft IS a quick-save: a leading `#` with
    /// something to save after it. A lone `#`, or `#!`, is prose and falls
    /// through to a normal turn.
    public static func isQuickSaveDraft(_ trimmedDraft: String) -> Bool {
        guard trimmedDraft.hasPrefix("#") else { return false }
        return !trimmedDraft.dropFirst().trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// The text a quick-save draft remembers: everything after the `#`.
    public static func memoryText(of trimmedDraft: String) -> String {
        trimmedDraft.dropFirst().trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func wrapping(_ memoryText: String) -> String {
        "\(openTag)\n\(memoryText)\n\(closeTag)"
    }

    /// The remembered text of a transcript entry, when it is a quick-save
    /// row. Nil for ordinary messages.
    public static func parse(_ content: String) -> String? {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix(openTag), let close = trimmed.range(of: closeTag) else { return nil }
        let inner = trimmed[trimmed.index(trimmed.startIndex, offsetBy: openTag.count)..<close.lowerBound]
        let text = inner.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }
}

/// Assembles the `## Memory` system-prompt section from a store's index.
///
/// Pure aside from the store it is handed, so both assemblers --
/// `AppModel.buildSystemPrompt` and `SubagentRunner.buildSystemPrompt` --
/// can call the same builder and a memory saved mid-session reaches every
/// kind of run. The guidance text is the port of Claude Code's memdir
/// behavioral prompt (`src/memdir/memdir.ts`), with the model writing
/// through this app's `memory` tool instead of raw file tools.
public enum MemoryPromptBuilder {
    /// The index budget, Claude Code's `MAX_ENTRYPOINT_LINES` /
    /// `MAX_ENTRYPOINT_BYTES`. An index past either is truncated, and the
    /// truncation says so in the injected text: a silently shortened index
    /// reads as complete memory rather than as a prompt to tidy the index.
    public static let maxIndexLines = 200
    public static let maxIndexBytes = 25_000

    /// Line- then byte-truncates an index, appending a self-describing
    /// warning when either budget fired. Byte truncation lands on the last
    /// complete line under the cap.
    public static func truncatedIndex(
        _ text: String, maxLines: Int = maxIndexLines, maxBytes: Int = maxIndexBytes
    ) -> (content: String, truncated: Bool) {
        var lines = SkillParser.normalizedLines(text)
        var reason: String?
        if lines.count > maxLines {
            lines = Array(lines.prefix(maxLines))
            reason = "too long (over \(maxLines) lines)"
        }
        var body = lines.joined(separator: "\n")
        if body.utf8.count > maxBytes {
            let utf8 = Array(body.utf8.prefix(maxBytes))
            // Cut THROUGH the last complete newline, so the warning below
            // starts on its own line and the surviving text ends at a row
            // boundary rather than mid-slug.
            if let lastNewline = utf8.lastIndex(of: UInt8(ascii: "\n")) {
                body = String(decoding: utf8[...lastNewline], as: UTF8.self)
            } else {
                body = String(decoding: utf8, as: UTF8.self)
            }
            reason = "too large (over \(maxBytes) bytes)"
        }
        guard let reason else {
            return (content: body, truncated: false)
        }
        return (
            content: body + "\n\n> WARNING: MEMORY.md is \(reason). Only part of it was loaded. "
                + "Keep index entries to one line under ~200 chars; move detail into topic files.",
            truncated: true
        )
    }

    /// The full section for one project: behavioral guidance plus the
    /// current index. Always emitted while memory is on, even with an empty
    /// index -- "your memory directory exists and is empty" is the state a
    /// first conversation needs to be told, or the model never saves
    /// anything and the feature never warms up.
    public static func section(store: MemoryStore, projectRoot: URL) -> String {
        let directory = store.directory(forProjectRoot: projectRoot).path
        let (index, _) = truncatedIndex(store.loadIndex(forProjectRoot: projectRoot))
        var lines: [String] = []
        lines.append("## Memory")
        lines.append("")
        lines.append("You have a persistent memory directory for this project at `\(directory)`. It exists; reach it with the `memory` tool rather than shell commands.")
        lines.append("")
        lines.append("MEMORY.md is the index of what you remember across conversations. Its current contents:")
        lines.append("")
        lines.append(index.isEmpty ? "(empty -- nothing is remembered yet)" : index)
        lines.append("")
        lines.append("To save something durable, call `memory` with action \"save\", a short kebab-case `name`, a one-line `description`, a `type`, and the `content`. Update an existing memory rather than creating a near-duplicate: action \"read\" shows one (no `name` reads the index), action \"forget\" removes one. Types: `user` is who the user is and how they prefer to work; `feedback` is guidance they have given, with why it matters and how to apply it; `project` is goals, constraints and decisions the code does not record; `reference` is a pointer to something outside this workspace.")
        lines.append("")
        lines.append("Do not save what the repository already records (code shape, git history, fix recipes), ephemeral task state, or secrets. Memories persist across conversations and can be stale; verify one against the current state before relying on it.")
        return lines.joined(separator: "\n")
    }
}
