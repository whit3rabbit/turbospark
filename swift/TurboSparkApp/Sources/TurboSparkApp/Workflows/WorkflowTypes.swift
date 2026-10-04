import CryptoKit
import Foundation

enum WorkflowRunState: String, Codable, CaseIterable, Sendable {
    case checking
    case awaitingApproval
    case declined
    case running
    case completed
    case failed
    case cancelled
    case interrupted
    case superseded
}

enum WorkflowErrorKind: String, Codable, CaseIterable, Sendable {
    case authoring
    case validation
    case sandboxRefusal
    case permissionDenied
    case resourceLimit
    case cancelled
    case modelFailure
}

struct WorkflowError: Error, Codable, Equatable, Sendable {
    var kind: WorkflowErrorKind
    var message: String
    var site: WorkflowSiteKey?
}

struct WorkflowStaticSite: Codable, Hashable, Sendable {
    var lane: String
    var siteIndex: Int
}

struct WorkflowSiteKey: Codable, Hashable, Sendable {
    var lane: String
    var siteIndex: Int
    var ordinal: Int

    init(lane: String, siteIndex: Int, ordinal: Int) {
        self.lane = lane
        self.siteIndex = siteIndex
        self.ordinal = ordinal
    }

    init(site: WorkflowStaticSite, ordinal: Int) {
        self.init(lane: site.lane, siteIndex: site.siteIndex, ordinal: ordinal)
    }

    var isValidJournalKey: Bool {
        !lane.isEmpty && ordinal >= 0
            && (siteIndex >= 0 || lane == "main" && WorkflowJournalSystemSites.contains(siteIndex))
    }
}

/// Engine records use reserved sites so they cannot collide with checked source calls.
enum WorkflowJournalSystemSites {
    static let cancellation = Int.min
    static let runFailure = Int.min + 1
    static let phase = Int.min + 2

    static func contains(_ index: Int) -> Bool {
        index == cancellation || index == runFailure || index == phase
    }
}

enum WorkflowSiteSequenceError: Error, Equatable, Sendable {
    case ordinalExhausted(site: WorkflowStaticSite)
}

/// Assigns deterministic, increasing ordinals to repeated calls at each static site.
struct WorkflowSiteKeySequence: Sendable {
    private var nextOrdinals: [WorkflowStaticSite: Int] = [:]

    mutating func next(lane: String, siteIndex: Int) throws -> WorkflowSiteKey {
        try next(for: WorkflowStaticSite(lane: lane, siteIndex: siteIndex))
    }

    mutating func next(for site: WorkflowStaticSite) throws -> WorkflowSiteKey {
        let ordinal = nextOrdinals[site, default: 0]
        guard ordinal < Int.max else {
            throw WorkflowSiteSequenceError.ordinalExhausted(site: site)
        }
        nextOrdinals[site] = ordinal + 1
        return WorkflowSiteKey(site: site, ordinal: ordinal)
    }
}

struct WorkflowRequestIdentity: Codable, Hashable, Sendable {
    var site: WorkflowSiteKey
    var inputHash: String

    init(site: WorkflowSiteKey, inputHash: String) {
        self.site = site
        self.inputHash = inputHash
    }

    static func make(site: WorkflowSiteKey, input: WorkflowCanonicalValue) throws -> Self {
        Self(site: site, inputHash: try WorkflowCanonicalSerialization.sha256Hex(input))
    }

    /// Actor asks depend on the transcript they continue, so changing prior work
    /// changes the request identity even when the new prompt is otherwise equal.
    static func make(
        site: WorkflowSiteKey,
        input: WorkflowCanonicalValue,
        priorActorTranscriptHash: String
    ) throws -> Self {
        let identityInput = WorkflowCanonicalValue.object([
            "input": input,
            "priorActorTranscriptHash": .string(priorActorTranscriptHash)
        ])
        return Self(
            site: site,
            inputHash: try WorkflowCanonicalSerialization.sha256Hex(identityInput))
    }
}

indirect enum WorkflowCanonicalValue: Codable, Equatable, Sendable {
    case null
    case boolean(Bool)
    case integer(Int64)
    case number(Double)
    case string(String)
    case array([WorkflowCanonicalValue])
    case object([String: WorkflowCanonicalValue])
}

enum WorkflowCanonicalSerializationError: Error, Equatable, Sendable {
    case unsupportedVersion(Int)
    case nonFiniteNumber
}

/// Canonical request bytes are domain-separated and versioned before they are hashed.
enum WorkflowCanonicalSerialization {
    static let currentVersion = 1

    static func encodedData(
        _ value: WorkflowCanonicalValue,
        version: Int = currentVersion
    ) throws -> Data {
        guard version == currentVersion else {
            throw WorkflowCanonicalSerializationError.unsupportedVersion(version)
        }

        var data = Data("turbospark.workflow.canonical-json.v1\n".utf8)
        try appendJSON(value, to: &data)
        return data
    }

    static func sha256Hex(
        _ value: WorkflowCanonicalValue,
        version: Int = currentVersion
    ) throws -> String {
        let digest = SHA256.hash(data: try encodedData(value, version: version))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func appendJSON(
        _ value: WorkflowCanonicalValue,
        to data: inout Data
    ) throws {
        switch value {
        case .null:
            data.append(contentsOf: "null".utf8)
        case .boolean(let value):
            data.append(contentsOf: (value ? "true" : "false").utf8)
        case .integer(let value):
            data.append(contentsOf: String(value).utf8)
        case .number(let value):
            guard value.isFinite else {
                throw WorkflowCanonicalSerializationError.nonFiniteNumber
            }
            data.append(contentsOf: String(value == 0 ? 0 : value).utf8)
        case .string(let value):
            appendJSONString(value.precomposedStringWithCanonicalMapping, to: &data)
        case .array(let values):
            data.append(0x5b)
            for index in values.indices {
                if index > values.startIndex {
                    data.append(0x2c)
                }
                try appendJSON(values[index], to: &data)
            }
            data.append(0x5d)
        case .object(let values):
            data.append(0x7b)
            let entries = values.sorted { left, right in
                left.key.precomposedStringWithCanonicalMapping.utf8.lexicographicallyPrecedes(
                    right.key.precomposedStringWithCanonicalMapping.utf8)
            }
            for index in entries.indices {
                if index > entries.startIndex {
                    data.append(0x2c)
                }
                appendJSONString(
                    entries[index].key.precomposedStringWithCanonicalMapping,
                    to: &data)
                data.append(0x3a)
                try appendJSON(entries[index].value, to: &data)
            }
            data.append(0x7d)
        }
    }

    private static func appendJSONString(_ value: String, to data: inout Data) {
        data.append(0x22)
        for byte in value.utf8 {
            switch byte {
            case 0x22:
                data.append(contentsOf: "\\\"".utf8)
            case 0x5c:
                data.append(contentsOf: "\\\\".utf8)
            case 0x00...0x1f:
                let hexDigits = Array("0123456789abcdef".utf8)
                data.append(contentsOf: [
                    0x5c,
                    0x75,
                    0x30,
                    0x30,
                    hexDigits[Int(byte >> 4)],
                    hexDigits[Int(byte & 0x0f)]
                ])
            default:
                data.append(byte)
            }
        }
        data.append(0x22)
    }
}

indirect enum WorkflowShapeValue: Codable, Equatable, Sendable {
    case string(enumValues: [String]?)
    case integer
    case number
    case boolean
    case null
    case array(item: WorkflowShapeValue)
    case object(fields: [WorkflowShapeField])
}

struct WorkflowShapeField: Codable, Equatable, Sendable {
    var name: String
    var required: Bool
    var value: WorkflowShapeValue
}

struct WorkflowResultShape: Codable, Equatable, Sendable {
    var fields: [WorkflowShapeField]
}

struct WorkflowFrozenArguments: Codable, Equatable, Sendable {
    private let values: [String: String]

    init(_ values: [String: String]) {
        self.values = values
    }

    subscript(name: String) -> String? {
        values[name]
    }

    var allValues: [String: String] {
        values
    }
}

struct WorkflowExecutableIdentity: Codable, Equatable, Sendable {
    var canonicalPath: String
    var sha256: String
}

enum WorkflowCommandArgumentKind: String, Codable, Sendable {
    case workspaceInputPath
    case allowedValue
    case boundedText
}

struct WorkflowCommandArgumentRule: Codable, Equatable, Sendable {
    var name: String
    var kind: WorkflowCommandArgumentKind
    var allowedValues: [String]?
    var maximumBytes: Int
}

enum WorkflowCommandArgSlot: Codable, Equatable, Sendable {
    case fixed(String)
    case dynamic(WorkflowCommandArgumentRule)
}

struct WorkflowCommandPin: Codable, Equatable, Sendable {
    var commandKey: String
    var executable: WorkflowExecutableIdentity
    var argvTemplate: [WorkflowCommandArgSlot]
    var workingDirectory: String
}

struct WorkflowActorSpec: Codable, Equatable, Sendable {
    var name: String
    var rolePrompt: String
}

struct WorkflowLaunchManifest: Codable, Equatable, Sendable {
    var phases: [String]
    var actors: [WorkflowActorSpec]
    var siteTable: [WorkflowStaticSite]
    var commandPins: [WorkflowCommandPin]

    init(
        phases: [String] = [],
        actors: [WorkflowActorSpec] = [],
        siteTable: [WorkflowStaticSite] = [],
        commandPins: [WorkflowCommandPin] = []
    ) {
        self.phases = phases
        self.actors = actors
        self.siteTable = siteTable
        self.commandPins = commandPins
    }
}

struct WorkflowRunDescriptor: Codable, Equatable, Sendable {
    var id: UUID
    var name: String
    var source: String
    var sourceHash: String
    var facadeVersion: Int
    let args: WorkflowFrozenArguments
    var manifest: WorkflowLaunchManifest
}

struct WorkflowArgumentDeclaration: Codable, Equatable, Sendable {
    var name: String
    var required: Bool
    var maximumUTF8Bytes: Int
}

enum WorkflowDefinitionScope: String, Codable, Sendable {
    case profile
}

struct WorkflowSavedDefinition: Codable, Equatable, Identifiable, Sendable {
    var id: UUID
    var name: String
    var scope: WorkflowDefinitionScope
    var source: String
    var declarations: [WorkflowArgumentDeclaration]
    var description: String
    var updatedAt: Date
}

protocol WorkflowSavedDefinitionStore: Sendable {
    func save(_ definition: WorkflowSavedDefinition) async throws
    func definition(id: UUID) async throws -> WorkflowSavedDefinition?
    func listDefinitions() async throws -> [WorkflowSavedDefinition]
}
