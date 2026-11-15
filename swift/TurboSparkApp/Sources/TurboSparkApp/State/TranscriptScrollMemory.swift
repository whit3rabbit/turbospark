import CoreGraphics
import Foundation

/// OC-04: where each open tab's transcript was scrolled to when the reader
/// left it. In memory only, like the transcript views it serves: nothing here
/// reaches disk.
///
/// A reader pinned to the live end is stored as `.followLatest`, never as an
/// offset: the transcript may have grown while the tab was away, and a stale
/// offset would strand a returning reader mid-stream.
struct TranscriptScrollMemory: Equatable {
    enum Position: Equatable {
        case followLatest
        /// Distance of the viewport's top from the document's top.
        case offset(CGFloat)
    }

    private(set) var positions: [UUID: Position] = [:]

    /// Remembers where `chatID` was left. `keeping` is the set of chats that
    /// still have a tab: anything else is dropped so the map cannot outgrow
    /// the tab cap. A chat that is not in `keeping` records nothing.
    mutating func record(
        chatID: UUID, pinned: Bool, offset: CGFloat?, keeping: Set<UUID>
    ) {
        positions = positions.filter { keeping.contains($0.key) }
        guard keeping.contains(chatID) else { return }
        if pinned {
            positions[chatID] = .followLatest
        } else if let offset, offset.isFinite {
            positions[chatID] = .offset(max(0, offset))
        }
        // Unpinned with no reading: keep whatever was there.
    }

    /// What to do when `chatID`'s transcript appears. nil means "do nothing
    /// special" (the caller follows the latest, as it always did). A pending
    /// reveal request wins over a remembered position: the reader just asked
    /// to see a particular row.
    func restore(for chatID: UUID, revealPending: Bool) -> Position? {
        guard !revealPending else { return nil }
        return positions[chatID]
    }

    mutating func forget(_ chatID: UUID) {
        positions[chatID] = nil
    }
}

extension TranscriptFollowRules {
    /// The offset a restore can actually reach for the current geometry.
    /// Lazily sized content may still be shorter than it will be, so the
    /// caller retries while this is below the target.
    static func reachableOffset(target: CGFloat, contentHeight: CGFloat, viewportHeight: CGFloat) -> CGFloat {
        min(max(0, target), max(0, contentHeight - viewportHeight))
    }
}
