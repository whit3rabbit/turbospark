import Foundation
import XCTest

/// Source-scan guard for VoiceOver announcements.
///
/// `AccessibilityNotification.post()` is an INSTANCE method. The spelling
/// `_ = AccessibilityNotification.Announcement.post(.init(message))`
/// compiles, because it evaluates the unapplied method with the new value as
/// its receiver and the `_ =` throws the resulting `() -> Void` away, and it
/// announces nothing. Every toast, error banner and generation start/finish
/// was silent to VoiceOver that way. The correct form is
/// `AccessibilityNotification.Announcement(message).post()`.
final class AccessibilityAnnouncementTests: XCTestCase {
    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Tests/TurboSparkAppTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // TurboSparkApp/
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    func testNoAnnouncementIsPostedThroughTheUnappliedMethod() throws {
        var problems: [String] = []
        var postedCount = 0
        let enumerator = FileManager.default.enumerator(
            at: sourcesRoot, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let source = try String(contentsOf: url, encoding: .utf8)
            for (index, line) in source.components(separatedBy: "\n").enumerated() {
                let code = line.components(separatedBy: "//").first ?? line
                if code.contains("Announcement.post(") {
                    problems.append("\(url.lastPathComponent):\(index + 1): \(line.trimmingCharacters(in: .whitespaces))")
                }
                if code.contains("AccessibilityNotification.Announcement(") && code.contains(").post()") {
                    postedCount += 1
                }
            }
        }
        XCTAssertTrue(
            problems.isEmpty,
            "use `AccessibilityNotification.Announcement(message).post()`; the static-looking "
                + "form announces nothing:\n" + problems.joined(separator: "\n"))
        // The scan must have seen the real call sites, or it proves nothing.
        XCTAssertGreaterThan(postedCount, 5, "the announcement call sites have vanished from the scan")
    }
}
