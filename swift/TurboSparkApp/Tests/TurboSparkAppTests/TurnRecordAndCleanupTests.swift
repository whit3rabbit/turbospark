import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Tier 4: an unfinished turn replayed as a finished one (state#65), a
/// cancelled command reported as a failure, and the two stores that deleted
/// or ran more than they meant to (state#64).
@MainActor
final class TurnRecordAndCleanupTests: XCTestCase {

    // MARK: - state#65: an unfinished turn is marked as one

    /// `finishCancelled` persists whatever the turn produced and records WHY
    /// in `stopReason`; the history builder never read it, so a reply cut off
    /// by an engine error was replayed on the next step as a completed
    /// assistant turn and the model built on half a sentence.
    func testAnInterruptedTurnCarriesANoteBackToTheModel() {
        let model = AppModel()
        var chat = AppChat(title: "interrupted")
        chat.messages = [
            AppChatMessage(role: .user, content: "explain"),
            AppChatMessage(
                role: .assistant, content: "Coastal wetlands red", stopReason: "cancelled"),
        ]
        model.chats = [chat]

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        let assistant = history.last { $0.role == .assistant }
        XCTAssertNotNil(assistant)
        XCTAssertTrue(assistant?.content.hasPrefix("Coastal wetlands red") ?? false)
        XCTAssertTrue(
            assistant?.content.contains("stopped by the user") ?? false,
            "The model must be told the turn did not finish.")
    }

    /// A turn that ended normally reads back byte for byte, or every reply in
    /// the transcript grows an annotation.
    func testACompletedTurnIsUnannotated() {
        let model = AppModel()
        var chat = AppChat(title: "finished")
        chat.messages = [
            AppChatMessage(role: .user, content: "hi"),
            AppChatMessage(role: .assistant, content: "hello", stopReason: "endOfTurn"),
        ]
        model.chats = [chat]

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        XCTAssertEqual(history.last?.content, "hello")
    }

    /// The three interrupted reasons each say what happened; a user turn is
    /// never annotated whatever its stop reason.
    func testTheNoteIsPerReasonAndAssistantOnly() {
        func assistant(_ reason: String?) -> AppChatMessage {
            AppChatMessage(role: .assistant, content: "x", stopReason: reason)
        }
        XCTAssertNotNil(AppModel.truncationNote(for: assistant("cancelled")))
        XCTAssertNotNil(AppModel.truncationNote(for: assistant("error")))
        XCTAssertNotNil(AppModel.truncationNote(for: assistant("context_overflow")))
        XCTAssertNil(AppModel.truncationNote(for: assistant("endOfTurn")))
        XCTAssertNil(AppModel.truncationNote(for: assistant(nil)))
        XCTAssertNil(
            AppModel.truncationNote(
                for: AppChatMessage(role: .user, content: "x", stopReason: "cancelled")))
    }

    // MARK: - state#64: the bare-alias metadata row is shared

    /// `key(for:path:)` prefers the path and `metadata(for:)` falls back to
    /// the bare alias, so deleting one model's metadata took the fallback row
    /// every OTHER model of that alias resolves through.
    func testRemovingOneModelsMetadataLeavesTheSharedFallbackAlone() {
        let store = ModelOrganizationStore.shared
        let alias = "shared-alias-\(UUID().uuidString.prefix(8))"
        defer {
            store.removeMetadata(for: alias)
            store.removeMetadata(for: alias, path: "/tmp/a.gturbo")
        }

        store.addTag("legacy", for: alias)
        store.addTag("first", for: alias, path: "/tmp/a.gturbo")

        store.removeMetadata(for: alias, path: "/tmp/a.gturbo")

        XCTAssertTrue(
            store.metadata(for: alias).tags.contains("legacy"),
            "A path-keyed delete must not take the bare-alias row with it.")
    }

    /// Deleting the bare-alias row itself still works: it is only spared when
    /// the call named a different, path-keyed model.
    func testRemovingTheBareAliasRowStillWorks() {
        let store = ModelOrganizationStore.shared
        let alias = "solo-alias-\(UUID().uuidString.prefix(8))"
        store.addTag("legacy", for: alias)
        store.removeMetadata(for: alias)
        XCTAssertFalse(store.metadata(for: alias).tags.contains("legacy"))
    }

    // MARK: - state#64: `executeSkill` gained the gates the tool already had
    //
    // **NOT ASSERTED HERE, AND THAT IS THE HONEST ANSWER.** `executeSkill`
    // resolves through `effectiveSkills`, which walks `~/.turbospark/skills`
    // and the selected project on the real disk -- `AppStorageRoot` covers
    // the app's JSON stores and not skill DISCOVERY (Gotcha 40's shape, one
    // store over). A case written against it would pass or fail on which
    // skills the developer happens to have installed, and a skip keyed on
    // "were any found" cannot tell an empty machine from a broken lookup
    // (Gotcha 36). The gate is two lines that mirror the `skill` tool's own,
    // which IS exercised; making this testable needs a discovery seam, which
    // is a refactor rather than a fix.
}
