import TurboSpark
import XCTest

@testable import TurboSparkApp

final class ServerDetachEventTests: XCTestCase {
    func testModelDetachedEventNamesTheServerAttachmentToRelease() {
        let event = ServerEvent.modelDetached(atMs: 123, model: "idle-model")
        XCTAssertEqual(AppModel.detachedServerModelID(from: event), "idle-model")
        XCTAssertNil(
            AppModel.detachedServerModelID(
                from: .modelAttached(atMs: 124, model: "other-model")))
    }

    func testReleaseRemovesOnlyTheDetachedServerOwnedSessionEntry() {
        var attachments = ["idle-model": 1, "still-serving": 2]
        AppModel.removeServerAttachment(id: "idle-model", from: &attachments)

        XCTAssertEqual(attachments, ["still-serving": 2])
    }
}
