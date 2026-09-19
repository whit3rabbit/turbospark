import Foundation

/// Large-paste handling for the composer (see `PromptPastePolicy` for the
/// pure half).
///
/// The flow on one draft change: `PromptComposerView`'s existing
/// `.onChange(of: model.promptText)` forwards the before/after values here;
/// a change that reads as a paste has its middle lifted into a
/// `Pasted Text.txt` attachment chip and the draft shrunk to the untouched
/// surroundings. The chip prices into the context ring through the ordinary
/// attachment estimate, previews through the ordinary attachment preview,
/// and inlines into the prompt at send time through the ordinary
/// attachment block -- no new send path.
extension AppModel {
    /// File name every converted paste carries. A `.txt` extension keeps
    /// `previewKind` on the text arm and `symbolName` on `doc.plaintext`.
    static let pasteAttachmentFileName = "Pasted Text.txt"

    /// Handles one draft change, converting it when it reads as a large
    /// paste. Cheap no-op on every ordinary change: the growth gate inside
    /// `splitPaste` rejects typing, autocomplete, deletions, and undo in
    /// constant time.
    func processLargePaste(previous: String, current: String) {
        guard promptWriteSuppressionDepth == 0, !current.isEmpty else { return }
        guard let split = PromptPastePolicy.splitPaste(previous: previous, current: current)
        else { return }

        // Mirror the send-time budget (state#36): the window minus what the
        // reply reserves. Without a session loaded the resolved window is
        // the 4,096 fallback, which still bounds the damage sensibly.
        let freeTokens = max(0, resolvedContextTokens - maxNewTokens)
        let text: String
        let wasTruncated: Bool
        switch PromptPastePolicy.fit(split.pasted, freeTokens: freeTokens) {
        case .fits:
            text = split.pasted
            wasTruncated = false
        case .truncated(let kept):
            text = kept
            wasTruncated = true
            showToast(
                String(
                    localized:
                        "Pasted text exceeded the \(freeTokens)-token context window; kept the first \(kept.count) characters.",
                    bundle: .module),
                style: .warning,
                duration: 6)
        case .noRoom:
            showToast(
                String(
                    localized:
                        "Not enough room in the \(resolvedContextTokens)-token context window for pasted text.",
                    bundle: .module),
                style: .warning,
                duration: 6)
            return
        }

        // Undoing a conversion re-inserts the same text through the editor,
        // which reads as a fresh paste; keeping the existing chip instead
        // of stacking an identical second one makes that round trip a
        // no-op on the attachment row.
        let isDuplicate =
            promptAttachments.last.map { !$0.isImage && $0.extractedText == text } ?? false
        if !isDuplicate {
            let attachment = AppPromptAttachment(
                fileName: Self.pasteAttachmentFileName,
                formatLabel: "Text",
                extractedText: text,
                wasTruncatedDuringExtraction: wasTruncated)
            // Mid-turn is legal here (the queue keeps accepting input while
            // a turn runs), so the interactive guard's refusal must not
            // fire; `appendPromptAttachmentDuringSubmission` is the same
            // tail without it, the exact path `MentionResolver` uses.
            if generating || submitting {
                appendPromptAttachmentDuringSubmission(attachment, toChatID: nil)
            } else {
                addPromptAttachment(attachment, toChatID: nil)
            }
        }

        // The shrink write only deletes, so the follow-up onChange can
        // never re-enter the conversion; the wrapper keeps the write
        // suppressed regardless.
        writePromptTextDirectly(split.draft)
        if !isDuplicate {
            showToast(
                String(
                    localized:
                        "Large paste attached as \"\(Self.pasteAttachmentFileName)\" (\(text.count) characters).",
                    bundle: .module),
                style: .info)
        }
    }

    /// Writes the draft through a programmatic path: history recall,
    /// queued-message restore, autocomplete acceptance, command and skill
    /// expansion, suggestion chips.
    ///
    /// These writes can legitimately insert thousands of characters, which
    /// is indistinguishable from a paste by content alone -- and converting
    /// one would corrupt the thing it belongs to (a history recall would no
    /// longer match `lastHistoryAppliedText`; a restored queued message
    /// would silently become an attachment). The depth counter makes them
    /// invisible to `processLargePaste`. Clearing writes (`promptText = ""`)
    /// need no wrapper: a deletion can never pass the growth gate.
    public func writePromptTextDirectly(_ newValue: String) {
        promptWriteSuppressionDepth += 1
        defer { promptWriteSuppressionDepth -= 1 }
        promptText = newValue
    }
}
