import Foundation

struct REPLLimits: Sendable, Equatable, Codable {
    var defaultTimeout: TimeInterval
    var maximumTimeout: TimeInterval
    var maximumOutputCharacters: Int
    var maximumImageBytes: Int
    var maximumRequestBytes: Int
    var footprintGrowthBudgetBytes: Int

    init(
        defaultTimeout: TimeInterval = 30,
        maximumTimeout: TimeInterval = 600,
        maximumOutputCharacters: Int = 30_000,
        maximumImageBytes: Int = 20 * 1_024 * 1_024,
        maximumRequestBytes: Int = 1_024 * 1_024,
        footprintGrowthBudgetBytes: Int = 512 * 1_024 * 1_024
    ) {
        self.defaultTimeout = defaultTimeout
        self.maximumTimeout = maximumTimeout
        self.maximumOutputCharacters = maximumOutputCharacters
        self.maximumImageBytes = maximumImageBytes
        self.maximumRequestBytes = maximumRequestBytes
        self.footprintGrowthBudgetBytes = footprintGrowthBudgetBytes
    }
}

struct REPLSessionConfiguration: Sendable, Equatable, Codable {
    var artifactDirectory: URL
}

struct REPLEmittedImage: Sendable, Equatable, Codable {
    var fileURL: URL
    var label: String?
}

struct REPLTextOutputEvent: Sendable, Equatable, Codable {
    enum Level: String, Sendable, Codable {
        case log
        case info
        case debug
        case warn
        case error
    }

    var level: Level
    var text: String
}

struct REPLCallResult: Sendable, Codable {
    enum Status: String, Sendable, Codable {
        case completed
        case failed
        case parseError
        case unsupportedSyntax
        case timedOut
        case cancelled
        case memoryLimit
        case busy
    }

    var status: Status
    var outputEvents: [REPLTextOutputEvent]
    var consoleText: String
    var errorText: String?
    var completionText: String?
    var images: [REPLEmittedImage]
    var truncated: Bool
    var sessionCreated: Bool
    var sessionReset: Bool
}
