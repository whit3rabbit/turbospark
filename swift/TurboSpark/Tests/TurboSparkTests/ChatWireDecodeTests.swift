import XCTest

@testable import TurboSpark

/// Decodes JSON in the exact shape crates/ffi/src/wire.rs serializes.
///
/// The fixtures are HAND-WRITTEN from the serde derives (not captured from a
/// Rust run): `WirePart::Image` has no skip_serializing_if, so an image part
/// always carries both `path` and `base64`, the unused one as null. Swift's
/// own encoder omits the absent key, so a Swift-only round trip never sees
/// this shape.
final class ChatWireDecodeTests: XCTestCase {

    private func decode(_ json: String) throws -> ChatMessage {
        try JSONDecoder().decode(ChatMessage.self, from: Data(json.utf8))
    }

    func testImagePathWithNullBase64Decodes() throws {
        let m = try decode(#"""
        {"role":"user","content":[
          {"type":"image","path":"/tmp/a.png","base64":null},
          {"type":"text","text":"What is this?"}]}
        """#)
        XCTAssertEqual(m, ChatMessage(role: .user, content: "What is this?",
                                      images: [.path("/tmp/a.png")]))
    }

    func testImageBase64WithNullPathDecodes() throws {
        let m = try decode(#"""
        {"role":"user","content":[{"type":"image","path":null,"base64":"QQ=="}]}
        """#)
        XCTAssertEqual(m, ChatMessage(role: .user, content: "", images: [.base64("QQ==")]))
    }

    func testTextPartAndBareStringContentDecode() throws {
        let parts = try decode(#"""
        {"role":"assistant","content":[{"type":"text","text":"hi"}]}
        """#)
        XCTAssertEqual(parts, .assistant("hi"))
        XCTAssertEqual(try decode(#"{"role":"tool","content":"out"}"#), .tool("out"))
    }

    func testUnknownPartTypeIsIgnoredRatherThanFatal() throws {
        let m = try decode(#"""
        {"role":"user","content":[{"type":"audio","path":"/x.wav","base64":null},
                                  {"type":"text","text":"t"}]}
        """#)
        XCTAssertEqual(m.content, "t")
        XCTAssertTrue(m.images.isEmpty)
    }

    func testEncodeOutputIsUnchanged() throws {
        let m = ChatMessage(role: .user, content: "q", images: [.path("/p.png")])
        let obj = try JSONSerialization.jsonObject(with: JSONEncoder().encode(m)) as! [String: Any]
        let parts = obj["content"] as! [[String: String]]
        XCTAssertEqual(parts, [["type": "image", "path": "/p.png"],
                               ["type": "text", "text": "q"]])
        let plain = try JSONEncoder().encode(ChatMessage.user("x"))
        let plainObj = try JSONSerialization.jsonObject(with: plain) as? [String: String]
        XCTAssertEqual(plainObj, ["role": "user", "content": "x"])
    }

    func testWindowFitOutcomeWithImageTurnDecodes() throws {
        let json = #"""
        {"retained":[
           {"role":"system","content":"sys"},
           {"role":"user","content":[
              {"type":"image","path":"/tmp/a.png","base64":null},
              {"type":"image","path":null,"base64":"QQ=="},
              {"type":"text","text":"compare"}]}],
         "measuredTokens":321,"removedTurnCount":2,"hasRoomForGeneration":true}
        """#
        let o = try JSONDecoder().decode(WindowFitOutcome.self, from: Data(json.utf8))
        XCTAssertEqual(o.measuredTokens, 321)
        XCTAssertEqual(o.removedTurnCount, 2)
        XCTAssertTrue(o.hasRoomForGeneration)
        XCTAssertEqual(o.retained[0], .system("sys"))
        XCTAssertEqual(o.retained[1].images, [.path("/tmp/a.png"), .base64("QQ==")])
        XCTAssertEqual(o.retained[1].content, "compare")
    }
}
