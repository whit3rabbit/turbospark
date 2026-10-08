import XCTest

@testable import TurboSpark

final class ToolWireTests: XCTestCase {
    private func json(_ value: some Encodable) throws -> [String: Any] {
        let data = try JSONEncoder().encode(value)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testAMessageWithoutToolsEncodesExactlyAsBefore() throws {
        let object = try json(ChatMessage.user("hi"))
        XCTAssertEqual(Set(object.keys), ["role", "content"], "no tool keys on a plain message")
        XCTAssertEqual(object["content"] as? String, "hi")
    }

    func testAnAssistantToolCallEncodesCamelCaseWithArgumentsAsText() throws {
        let call = GenerationToolCall(id: "toolu_0", name: "get_weather", argumentsJSON: #"{"city":"Oslo"}"#)
        let object = try json(ChatMessage.assistant("", toolCalls: [call]))
        let calls = try XCTUnwrap(object["toolCalls"] as? [[String: Any]])
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0]["id"] as? String, "toolu_0")
        XCTAssertEqual(calls[0]["name"] as? String, "get_weather")
        XCTAssertEqual(calls[0]["arguments"] as? String, #"{"city":"Oslo"}"#)
    }

    func testAToolResultCarriesItsCallIdAndName() throws {
        let object = try json(ChatMessage.tool("12C", toolCallId: "toolu_0", name: "get_weather"))
        XCTAssertEqual(object["role"] as? String, "tool")
        XCTAssertEqual(object["toolCallId"] as? String, "toolu_0")
        XCTAssertEqual(object["name"] as? String, "get_weather")
    }

    func testToolFieldsSurviveAnEngineRoundTrip() throws {
        let wire = #"""
        {"role":"assistant","content":"",
         "toolCalls":[{"id":"toolu_1","name":"f","arguments":{"n":2,"ok":true}}]}
        """#
        let message = try JSONDecoder().decode(ChatMessage.self, from: Data(wire.utf8))
        XCTAssertEqual(message.toolCalls.map(\.name), ["f"])
        XCTAssertEqual(message.toolCalls[0].argumentsJSON, #"{"n":2,"ok":true}"#)
        // Re-encoding keeps the call, so a value fed back through the engine
        // (fitWindow's retained turns) still replays it.
        let again = try JSONDecoder().decode(
            ChatMessage.self, from: JSONEncoder().encode(message))
        XCTAssertEqual(again, message)
    }

    func testToolOptionsEncodeAsNameDescriptionAndASchemaObject() throws {
        var options = GenerateOptions()
        options.tools = [
            try ToolSpec(
                name: "get_weather", description: "Weather",
                parametersJSON: #"{"type":"object","properties":{"city":{"type":"string"}}}"#)
        ]
        let object = try json(options)
        let tools = try XCTUnwrap(object["tools"] as? [[String: Any]])
        XCTAssertEqual(tools[0]["name"] as? String, "get_weather")
        XCTAssertEqual(tools[0]["description"] as? String, "Weather")
        let schema = try XCTUnwrap(tools[0]["parameters"] as? [String: Any], "an object, not text")
        XCTAssertEqual(schema["type"] as? String, "object")
    }

    func testNoToolsEncodesAnEmptyArray() throws {
        XCTAssertEqual((try json(GenerateOptions())["tools"] as? [Any])?.count, 0)
    }

    func testJSONValueRoundTripsEveryKindAndInvalidTextThrows() throws {
        let text = #"{"a":[1,2.5,"x",true,null],"b":{"c":"d"}}"#
        let value = try JSONValue(jsonString: text)
        XCTAssertEqual(value.jsonString, text)
        XCTAssertThrowsError(try JSONValue(jsonString: "{nope"))
    }
}
