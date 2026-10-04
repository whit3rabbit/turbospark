import XCTest

@testable import TurboSparkApp

final class WorkflowTypesTests: XCTestCase {
    private func canonicalInput(argumentPairs: [(String, String)]) -> WorkflowCanonicalValue {
        .object([
            "prompt": .string("summarize"),
            "arguments": .object(Dictionary(uniqueKeysWithValues: argumentPairs.map {
                ($0.0, .string($0.1))
            }))
        ])
    }

    func testCanonicalRequestHashSurvivesSimulatedRestart() throws {
        let original = canonicalInput(argumentPairs: [("zeta", "last"), ("alpha", "first")])
        let persisted = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(WorkflowCanonicalValue.self, from: persisted)

        let firstHash = try WorkflowCanonicalSerialization.sha256Hex(original)
        let restartedHash = try WorkflowCanonicalSerialization.sha256Hex(restored)

        XCTAssertEqual(firstHash, restartedHash)
        XCTAssertEqual(firstHash.count, 64)
        let versionPrefix = Data("turbospark.workflow.canonical-json.v1\n".utf8)
        XCTAssertEqual(
            try WorkflowCanonicalSerialization.encodedData(original).prefix(versionPrefix.count),
            versionPrefix)
        let json = try WorkflowCanonicalSerialization.encodedData(original).dropFirst(
            versionPrefix.count)
        XCTAssertEqual(
            String(decoding: json, as: UTF8.self),
            "{\"arguments\":{\"alpha\":\"first\",\"zeta\":\"last\"},\"prompt\":\"summarize\"}")
        XCTAssertEqual(
            firstHash,
            "21bb8a6d9a4c8bc271314cf699aff507654ea9ff032682b3b421e6bcc3512f3d")
    }

    func testCanonicalRequestHashIgnoresDictionaryInsertionOrder() throws {
        let first = canonicalInput(argumentPairs: [("a", "1"), ("z", "9")])
        let second = canonicalInput(argumentPairs: [("z", "9"), ("a", "1")])

        XCTAssertEqual(
            try WorkflowCanonicalSerialization.encodedData(first),
            try WorkflowCanonicalSerialization.encodedData(second))
        XCTAssertEqual(
            try WorkflowCanonicalSerialization.sha256Hex(first),
            try WorkflowCanonicalSerialization.sha256Hex(second))
    }

    func testPriorActorTranscriptHashParticipatesInRequestIdentity() throws {
        let site = WorkflowSiteKey(
            lane: "writer",
            siteIndex: 3,
            ordinal: 0)
        let input = canonicalInput(argumentPairs: [("topic", "continuity")])
        let first = try WorkflowRequestIdentity.make(
            site: site,
            input: input,
            priorActorTranscriptHash: "transcript-a")
        let restarted = try WorkflowRequestIdentity.make(
            site: site,
            input: input,
            priorActorTranscriptHash: "transcript-a")
        let changedHistory = try WorkflowRequestIdentity.make(
            site: site,
            input: input,
            priorActorTranscriptHash: "transcript-b")

        XCTAssertEqual(first, restarted)
        XCTAssertNotEqual(first.inputHash, changedHistory.inputHash)
    }

    func testUnknownCanonicalizationVersionIsRejected() {
        XCTAssertThrowsError(
            try WorkflowCanonicalSerialization.encodedData(.string("input"), version: 2)) { error in
            XCTAssertEqual(
                error as? WorkflowCanonicalSerializationError,
                .unsupportedVersion(2))
        }
    }

    func testCanonicalEncodingNormalizesEquivalentStringsAndNegativeZero() throws {
        let composed = WorkflowCanonicalValue.string("\u{00e9}")
        let decomposed = WorkflowCanonicalValue.string("e\u{0301}")

        XCTAssertEqual(composed, decomposed)
        XCTAssertEqual(
            try WorkflowCanonicalSerialization.sha256Hex(composed),
            try WorkflowCanonicalSerialization.sha256Hex(decomposed))
        XCTAssertEqual(
            try WorkflowCanonicalSerialization.sha256Hex(.number(-0.0)),
            try WorkflowCanonicalSerialization.sha256Hex(.number(0.0)))
    }

    func testCanonicalEncodingEscapesControlBytesAndRejectsNonFiniteNumbers() throws {
        let versionPrefix = Data("turbospark.workflow.canonical-json.v1\n".utf8)
        let encoded = try WorkflowCanonicalSerialization.encodedData(.string("line\nnext"))
        let json = encoded.dropFirst(versionPrefix.count)

        XCTAssertEqual(String(decoding: json, as: UTF8.self), "\"line\\u000anext\"")
        XCTAssertThrowsError(
            try WorkflowCanonicalSerialization.encodedData(.number(.infinity))) { error in
            XCTAssertEqual(error as? WorkflowCanonicalSerializationError, .nonFiniteNumber)
        }
    }

    func testFanOutSiteCallsReceiveUniqueIncreasingOrdinals() throws {
        var sequence = WorkflowSiteKeySequence()
        let fanOutSite = WorkflowStaticSite(lane: "researcher", siteIndex: 4)
        let keys = try (0..<100).map { _ in try sequence.next(for: fanOutSite) }

        XCTAssertEqual(Set(keys).count, 100)
        XCTAssertEqual(keys.map(\.ordinal), Array(0..<100))
        XCTAssertEqual(try sequence.next(for: fanOutSite).ordinal, 100)
        XCTAssertEqual(try sequence.next(lane: "main", siteIndex: 4).ordinal, 0)
    }

    func testWorkflowTypesRoundTripThroughCodable() throws {
        let shape = WorkflowResultShape(fields: [
            WorkflowShapeField(
                name: "verdict",
                required: true,
                value: .string(enumValues: ["accept", "revise"])),
            WorkflowShapeField(
                name: "sources",
                required: true,
                value: .array(item: .string(enumValues: nil)))
        ])
        let args = WorkflowFrozenArguments(["topic": "local models"])
        let descriptor = WorkflowRunDescriptor(
            id: UUID(uuidString: "9D26B99B-37DF-4F6C-82F8-C4B4B91DC606")!,
            name: "research",
            source: "async function workflow() {}",
            sourceHash: "source-hash",
            facadeVersion: 1,
            args: args,
            manifest: WorkflowLaunchManifest(
                phases: ["research"],
                actors: [WorkflowActorSpec(name: "researcher", rolePrompt: "Research the topic")],
                siteTable: [WorkflowStaticSite(lane: "researcher", siteIndex: 2)]))
        let definition = WorkflowSavedDefinition(
            id: UUID(uuidString: "DE3978C4-E1AA-40CD-9B04-E3E851215391")!,
            name: "saved research",
            scope: .profile,
            source: descriptor.source,
            declarations: [WorkflowArgumentDeclaration(
                name: "topic", required: true, maximumUTF8Bytes: 4_096)],
            description: "Research a supplied topic",
            updatedAt: Date(timeIntervalSince1970: 1_790_000_000))

        XCTAssertEqual(try roundTrip(shape), shape)
        XCTAssertEqual(try roundTrip(descriptor), descriptor)
        XCTAssertEqual(try roundTrip(definition), definition)
        XCTAssertEqual(args["topic"], "local models")
        XCTAssertNil(args["missing"])
    }

    func testErrorKindsAndRunStatesStayCodableAndComplete() throws {
        XCTAssertEqual(WorkflowRunState.allCases.count, 9)
        XCTAssertEqual(WorkflowErrorKind.allCases.count, 7)

        let error = WorkflowError(
            kind: .permissionDenied,
            message: "approval declined",
            site: WorkflowSiteKey(lane: "main", siteIndex: 3, ordinal: 7))
        XCTAssertEqual(try roundTrip(error), error)
        XCTAssertEqual(try roundTrip(WorkflowRunState.interrupted), .interrupted)
    }

    private func roundTrip<Value: Codable>(_ value: Value) throws -> Value {
        try JSONDecoder().decode(Value.self, from: JSONEncoder().encode(value))
    }
}
