import Foundation

/// A request-only transcript notice for tool output shortened by microcompact.
struct CompactionBoundaryEvent: Equatable, Identifiable, Sendable {
    var id = UUID()
    var chatID: UUID
    var rowsCleared: Int
    var estimatedTokensSaved: Int
}
