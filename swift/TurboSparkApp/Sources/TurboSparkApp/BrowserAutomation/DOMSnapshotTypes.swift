import Foundation
import CoreFoundation
import WebKit

public struct DOMSnapshotBounds: Codable, Equatable, Sendable {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

public struct DOMSnapshotNode: Codable, Equatable, Sendable {
    public let role: String
    public let name: String
    public let parentIndex: Int?
    public let bounds: DOMSnapshotBounds?
    public let reference: String?

    public init(
        role: String,
        name: String,
        parentIndex: Int?,
        bounds: DOMSnapshotBounds?,
        reference: String?
    ) {
        self.role = role
        self.name = name
        self.parentIndex = parentIndex
        self.bounds = bounds
        self.reference = reference
    }
}

public struct DOMSnapshotEnvelope: Codable, Equatable, Sendable {
    public let version: String
    public let generation: UUID
    public let nodes: [DOMSnapshotNode]
    public let truncated: Bool

    public init(version: String, generation: UUID, nodes: [DOMSnapshotNode], truncated: Bool) {
        self.version = version
        self.generation = generation
        self.nodes = nodes
        self.truncated = truncated
    }
}

public enum DOMSnapshotServiceError: Error, Equatable {
    case inactiveDocument
    case rejectedMessage
    case invalidSnapshot
    case invalidLocator
    case targetNotFound
    case locatorLimitExceeded
    case invalidAction
    case actionRejected
    case unavailable
}

public enum DOMSnapshotLocator: Sendable, Equatable {
    case reference(String)
    case role(role: String, name: String)
    case visibleText(String)
    case cssSelector(String)
}

public enum DOMSnapshotScrollDirection: String, Sendable, Equatable {
    case up
    case down
    case left
    case right
}

public enum DOMSnapshotAction: Sendable, Equatable {
    case click
    case type(text: String, submit: Bool)
    case pressKey(key: String)
    case scroll(direction: DOMSnapshotScrollDirection, amount: Int)
}

public struct DOMSnapshotResolvedTarget: Sendable, Equatable {
    public let reference: String
    public let isAmbiguous: Bool

    public init(reference: String, isAmbiguous: Bool) {
        self.reference = reference
        self.isAmbiguous = isAmbiguous
    }
}

public struct DOMSnapshotActionOutcome: Sendable, Equatable {
    public let reference: String
    public let isAmbiguous: Bool
    public let didSubmit: Bool

    public init(reference: String, isAmbiguous: Bool, didSubmit: Bool) {
        self.reference = reference
        self.isAmbiguous = isAmbiguous
        self.didSubmit = didSubmit
    }
}

public struct DOMSnapshotPickCandidate: Codable, Equatable, Sendable {
    public let version: String
    public let generation: UUID
    public let reference: String
    public let role: String
    public let name: String
    public let bounds: DOMSnapshotBounds

    public init(
        version: String = "ts_element_pick_v1",
        generation: UUID,
        reference: String,
        role: String,
        name: String,
        bounds: DOMSnapshotBounds
    ) {
        self.version = version
        self.generation = generation
        self.reference = reference
        self.role = role
        self.name = name
        self.bounds = bounds
    }
}

