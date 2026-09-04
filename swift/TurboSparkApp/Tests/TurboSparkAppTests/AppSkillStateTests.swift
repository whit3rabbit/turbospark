import XCTest

@testable import TurboSparkApp

/// These mirror the cases in `scripts/skill_state_probe.py`'s `--selfcheck`,
/// which is the harness the measurement in `docs/SKILL_STATE.md` was taken
/// with. The probe's semantics are the evidence, so a divergence here makes
/// those numbers stop describing this code.
///
/// The probe's own mutation testing found one gap worth repeating: nothing
/// exercised merge's null-deletion branch, because no control agent emits a
/// null. `mergeNullDeletesAField` and its siblings are that gap closed.
final class AppSkillStateTests: XCTestCase {

    // MARK: - state#91 and state#92: the prompt and the badge

    /// The patch rules are the model's only description of what `apply` does,
    /// and one of them was the opposite of the truth: it said a MAP you
    /// include replaces the stored one, where `apply` merges it entry by
    /// entry. A model following that has no way to delete an entry at all, so
    /// `files` only ever grew -- to `maxRenderedBytes`, after which every
    /// patch was rejected for a limit the instructions said could not be
    /// avoided.
    @MainActor
    func testThePatchRulesDescribeTheMergeSemanticsApplyActuallyHas() {
        let prompt = AppModel().skillStateProtocol()
        XCTAssertTrue(
            prompt.contains("MERGES entry by entry"),
            "The prompt must describe merge, because that is what `apply` does.")
        XCTAssertTrue(
            prompt.contains("null to delete it"),
            "And it must say how to remove an entry, or nothing can ever shrink.")
        XCTAssertFalse(
            prompt.contains("A list or a map you include REPLACES"),
            "The old sentence is the defect; a map does not replace.")

        // The behaviour the sentence now describes.
        var s = state(["files": .object(["a.swift": .string("read"), "b.swift": .string("read")])])
        s.apply(patch: ["files": .object(["b.swift": .null])])
        XCTAssertEqual(
            s.fields, ["files": .object(["a.swift": .string("read")])],
            "A named entry set to null is dropped and the rest are kept.")
    }

    /// state#92: a `<state_patch>` block whose body is not JSON fell into the
    /// no-patch arm and CLEARED the badge -- so the one failure a user cannot
    /// see in the transcript (the block is stripped before display) was also
    /// the one that reported nothing.
    @MainActor
    func testAnUnparseablePatchRaisesTheBadgeRatherThanClearingIt() {
        let appModel = AppModel()
        var project = AppProject(name: "P", rootDirectoryPath: "/tmp")
        project.skillStateEnabled = true
        let chat = AppChat(projectID: project.id, title: "one")
        appModel.projects = [project]
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let stripped = appModel.applySkillStatePatch(
            from: "working on it\n<state_patch>\nnot json at all\n</state_patch>",
            chatIndex: 0, project: project)

        XCTAssertFalse(
            stripped.contains("state_patch"), "The block is still removed from the reply.")
        XCTAssertNotNil(
            appModel.skillStateLastError,
            "A patch that would not parse is an error, not an absence.")

        // And a turn with no block at all still clears it (state#58).
        appModel.applySkillStatePatch(from: "just prose", chatIndex: 0, project: project)
        XCTAssertNil(
            appModel.skillStateLastError,
            "A clean turn must still clear the badge, or one bad patch lights it forever.")
    }

    // MARK: - merge (RFC 7386)

    private func state(_ pairs: [String: AppJSONValue]) -> AppSkillState {
        AppSkillState(fields: pairs)
    }

    func testMergeNullDeletesAField() {
        var s = state(["goal": .string("ship it"), "facts": .array([.string("a")])])
        s.apply(patch: ["facts": .null])
        XCTAssertEqual(s.fields, ["goal": .string("ship it")])
    }

    func testMergeNullDeletesOnlyItsOwnKey() {
        var s = state(["goal": .string("g"), "facts": .array([.string("a")])])
        s.apply(patch: ["goal": .null])
        XCTAssertEqual(s.fields, ["facts": .array([.string("a")])])
    }

    func testMergeReplacesArraysWholesaleRatherThanAppending() {
        var s = state(["facts": .array([.string("a"), .string("b")])])
        s.apply(patch: ["facts": .array([.string("c")])])
        XCTAssertEqual(s.fields["facts"], .array([.string("c")]))
    }

    func testMergeOfAnEmptyPatchChangesNothing() {
        let before = state(["goal": .string("g"), "facts": .array([.string("a")])])
        XCTAssertEqual(before.applying(patch: [:]), before)
    }

    func testMergeLeavesAbsentKeysUntouched() {
        var s = state(["goal": .string("g"), "facts": .array([.string("a")])])
        s.apply(patch: ["goal": .string("h")])
        XCTAssertEqual(s.fields["facts"], .array([.string("a")]))
    }

    /// A map field merges per entry, so recording one more file does not wipe
    /// the others. This is the one place the merge is NOT a wholesale replace,
    /// and the prompt says so.
    func testMergeOfAMapAddsAnEntryWithoutDroppingTheRest() {
        var s = state(["files": .object(["a.swift": .string("read")])])
        s.apply(patch: ["files": .object(["b.swift": .string("edited")])])
        XCTAssertEqual(
            s.fields["files"],
            .object(["a.swift": .string("read"), "b.swift": .string("edited")]))
    }

    func testMergeOfANullMapEntryDropsThatEntryAlone() {
        var s = state([
            "files": .object(["a.swift": .string("read"), "b.swift": .string("edited")])
        ])
        s.apply(patch: ["files": .object(["a.swift": .null])])
        XCTAssertEqual(s.fields["files"], .object(["b.swift": .string("edited")]))
    }

    // MARK: - validation

    func testValidPatchShapes() {
        let cases: [[String: AppJSONValue]] = [
            [:],
            ["goal": .string("ship it")],
            ["facts": .array([.string("a")])],
            ["files": .object(["a.swift": .string("read")])],
            ["goal": .null],
            ["files": .object(["a.swift": .null])],
        ]
        for patch in cases {
            XCTAssertTrue(
                AppSkillStatePatch.validate(patch).isEmpty,
                "expected valid: \(patch)")
        }
    }

    func testInvalidPatchShapesAreRejected() {
        let cases: [(String, [String: AppJSONValue])] = [
            ("unknown field", ["inventory": .object([:])]),
            ("goal must be a string", ["goal": .array([.string("a")])]),
            ("facts must be an array", ["facts": .string("a")]),
            ("facts must hold strings", ["facts": .array([.number(3)])]),
            ("files must be an object", ["files": .array([.string("a")])]),
            ("a file entry must be a string", ["files": .object(["a": .number(1)])]),
        ]
        for (why, patch) in cases {
            XCTAssertFalse(
                AppSkillStatePatch.validate(patch).isEmpty,
                "expected rejection: \(why)")
        }
    }

    // MARK: - extraction

    func testExtractsATaggedPatch() {
        let text = "Doing it.\n<state_patch>\n{\"goal\": \"ship\"}\n</state_patch>"
        XCTAssertEqual(AppSkillStatePatch.extract(from: text), ["goal": .string("ship")])
    }

    /// The measured formatting hazard: gemma4 fenced 92% of its replies where
    /// gptoss fenced none, and rescue is what took all three installs to a
    /// 1.00 valid-patch rate.
    func testExtractsAFencedPatchInsideTheTag() {
        let text = "<state_patch>\n```json\n{\"goal\": \"ship\"}\n```\n</state_patch>"
        XCTAssertEqual(AppSkillStatePatch.extract(from: text), ["goal": .string("ship")])
    }

    func testExtractsAPatchNestedUnderAPatchKey() {
        let text = "{\"reasoning\": \"noted\", \"patch\": {\"goal\": \"ship\"}}"
        XCTAssertEqual(AppSkillStatePatch.extract(from: text), ["goal": .string("ship")])
    }

    func testBracesInsideAStringDoNotEndTheScan() {
        let text = "<state_patch>{\"goal\": \"use }{ carefully\"}</state_patch>"
        XCTAssertEqual(
            AppSkillStatePatch.extract(from: text), ["goal": .string("use }{ carefully")])
    }

    /// A turn that changes nothing carries no patch, and that is the correct
    /// answer rather than an error. 14 of 14 no-op events per probe run were
    /// handled this way on every install.
    func testProseWithNoPatchYieldsNothing() {
        XCTAssertNil(AppSkillStatePatch.extract(from: "I have finished the task."))
    }

    /// An untagged bare object is far more likely to be a tool call than a
    /// patch, so it must NOT be swallowed as one.
    func testAnUntaggedToolCallIsNotMistakenForAPatch() {
        let text = "{\"name\": \"run_command\", \"arguments\": {\"command\": \"ls\"}}"
        XCTAssertNil(AppSkillStatePatch.extract(from: text))
    }

    // MARK: - rendering

    func testRenderingIsCanonicalSoUnchangedFieldsDoNotMove() {
        let a = state(["goal": .string("g"), "facts": .array([.string("x")])])
        let b = state(["facts": .array([.string("x")]), "goal": .string("g")])
        XCTAssertEqual(a.rendered, b.rendered)
        XCTAssertTrue(a.rendered.contains("\"facts\""))
    }

    func testEmptyStateRendersAsAnEmptyObject() {
        XCTAssertEqual(AppSkillState().rendered, "{}")
    }

    // MARK: - persistence

    /// Gotcha 13: a chat saved before this field existed must still decode.
    func testAChatSavedBeforeSkillStateExistedStillDecodes() throws {
        let json = """
            {"id":"\(UUID().uuidString)","title":"t","draft":"","messages":[]}
            """
        let chat = try JSONDecoder().decode(AppChat.self, from: Data(json.utf8))
        XCTAssertNil(chat.skillState)
    }

    func testAProjectSavedBeforeSkillStateExistedDefaultsToOff() throws {
        let json = "{\"name\":\"p\"}"
        let project = try JSONDecoder().decode(AppProject.self, from: Data(json.utf8))
        XCTAssertFalse(project.skillStateEnabled)
    }

    func testStateRoundTripsThroughTheArchiveEncoding() throws {
        var chat = AppChat(title: "t")
        chat.skillState = state(["goal": .string("g"), "files": .object(["a": .string("b")])])
        let data = try JSONEncoder().encode(chat)
        let back = try JSONDecoder().decode(AppChat.self, from: data)
        XCTAssertEqual(back.skillState, chat.skillState)
    }

    // MARK: - the schema is one source

    /// The prompt and the validator must agree, or the model is told to emit
    /// something the runtime then rejects. Both read `AppSkillStateSchema`, and
    /// this asserts the description actually names every validated field.
    func testEveryValidatedFieldAppearsInThePromptDescription() {
        let described = AppSkillStateSchema.promptDescription
        for field in AppSkillStateSchema.fields {
            XCTAssertTrue(
                described.contains("\"\(field.name)\""),
                "\(field.name) is validated but never described to the model")
        }
    }
}
