import XCTest
@testable import TurboSparkApp

/// The transcript's artifact chip shows two derived facts: a call that is
/// still writing a file that will become an artifact, and the revision of a
/// file written more than once. Both come only from state the app already
/// holds (`AppToolCall.status`, the result list, `AppArtifact.revision`).
final class ArtifactChipStateTests: XCTestCase {
    private let chatID = UUID()

    private func call(
        _ name: String, _ arguments: [String: String], status: AppToolCallStatus = .running
    ) -> AppToolCall {
        AppToolCall(name: name, arguments: arguments, status: status, category: .fileWrite)
    }

    func testRunningMarkdownWriteWithNoResultIsWriting() {
        let write = call("write_file", ["path": "docs/plan.md", "content": "x"])
        let rows = ArtifactChipState.writing(calls: [write], results: [], artifacts: [])
        XCTAssertEqual(rows, [ArtifactChipState.Writing(id: write.id, fileName: "plan.md")])
    }

    func testEditAliasAndHtmlCount() {
        let edit = call("edit_file", ["file_path": "site/index.HTML", "old_string": "a", "new_string": "b"])
        let rows = ArtifactChipState.writing(calls: [edit], results: [], artifacts: [])
        XCTAssertEqual(rows.map(\.fileName), ["index.HTML"])
    }

    /// The registrar only promotes markdown and HTML writes, so a source
    /// edit must not show a chip that never turns into an artifact.
    func testSourceFileWriteIsNotWriting() {
        let write = call("write_file", ["path": "src/lib.rs", "content": "x"])
        XCTAssertTrue(ArtifactChipState.writing(calls: [write], results: [], artifacts: []).isEmpty)
    }

    func testNonRunningStatusesAreNotWriting() {
        for status in [AppToolCallStatus.pendingApproval, .completed, .denied, .failed] {
            let write = call("write_file", ["path": "a.md"], status: status)
            XCTAssertTrue(
                ArtifactChipState.writing(calls: [write], results: [], artifacts: []).isEmpty,
                "\(status)")
        }
    }

    func testCallWithResultIsNotWriting() {
        let write = call("write_file", ["path": "a.md"])
        let result = AppToolResult(callID: write.id, output: "ok")
        XCTAssertTrue(ArtifactChipState.writing(calls: [write], results: [result], artifacts: []).isEmpty)
    }

    /// Once the artifact row anchors on this call, the real chip takes over.
    func testCallAlreadyAnchoringAnArtifactIsNotWriting() {
        let write = call("write_file", ["path": "/tmp/a.md"])
        let artifact = AppArtifact(
            chatID: chatID, path: "/tmp/a.md", title: "a.md", origin: .fileWrite, toolCallID: write.id)
        XCTAssertTrue(ArtifactChipState.writing(calls: [write], results: [], artifacts: [artifact]).isEmpty)
    }

    func testUnknownToolOrMissingOrConflictingPathIsNotWriting() {
        let read = call("read_file", ["path": "a.md"])
        let noPath = call("write_file", ["content": "x"])
        let conflict = call("write_file", ["path": "a.md", "file_path": "b.md"])
        XCTAssertTrue(
            ArtifactChipState.writing(calls: [read, noPath, conflict], results: [], artifacts: []).isEmpty)
    }

    func testRevisionBadgeOnlyAfterARewrite() {
        XCTAssertNil(ArtifactChipState.revisionBadge(revision: 1))
        XCTAssertNil(ArtifactChipState.revisionBadge(revision: 0))
        XCTAssertEqual(ArtifactChipState.revisionBadge(revision: 2), 2)
        XCTAssertEqual(ArtifactChipState.revisionBadge(revision: 7), 7)
    }

    /// The revision the badge reads is the one `upsert` bumps on a rewrite.
    func testUpsertedRewriteCarriesBadge() {
        var artifacts: [AppArtifact] = []
        let first = AppArtifact(chatID: chatID, path: "/tmp/p.md", title: "p.md", origin: .fileWrite)
        AppArtifact.upsert(first, into: &artifacts)
        XCTAssertNil(ArtifactChipState.revisionBadge(revision: artifacts[0].revision))
        AppArtifact.upsert(first, into: &artifacts)
        AppArtifact.upsert(first, into: &artifacts)
        XCTAssertEqual(ArtifactChipState.revisionBadge(revision: artifacts[0].revision), 3)
    }
}
