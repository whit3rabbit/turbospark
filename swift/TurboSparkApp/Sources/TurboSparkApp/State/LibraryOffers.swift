import Foundation

/// An offer to add a library found on a drive that just connected (requirement
/// 4.3). Published state only: the banner and the toast text are UI (task 6.6).
///
/// `name` is raw and untrusted, because it came from a marker on a drive anyone
/// could have written; show it through `UntrustedText.display`.
struct LibraryOffer: Identifiable, Equatable, Sendable {
    var id: String { markerId }

    /// The mounted volume's path. `TurboSparkLibraries.add` takes it as is: it
    /// adopts the standard `TurboSpark` folder at the top of the volume.
    let volumeRoot: String
    /// The library's id from its marker. Dismissals are keyed on it, so a
    /// dismissed library stays dismissed whatever the drive is called next time.
    let markerId: String
    let name: String
    let modelCount: Int
    /// A library a newer TurboSpark wrote: it can be listed but not changed.
    let readOnly: Bool
}

/// The offers waiting for an answer: one on screen, the rest in line.
///
/// **THE RULE FOR A SECOND LIBRARY: IT QUEUES.** Requirement 4.3 says an offer
/// stays visible until the user adds or dismisses it, so a second unknown
/// library must not replace the one on screen. It waits its turn and appears
/// when the first is answered. An id already on screen or in line is not added
/// again (a drive unplugged and plugged back in does not queue behind itself).
///
/// Pure value logic with no storage, so the rule is pinned without a drive.
struct LibraryOfferQueue: Equatable {
    private(set) var pending: LibraryOffer?
    private(set) var queued: [LibraryOffer] = []

    func contains(markerId: String) -> Bool {
        pending?.markerId == markerId || queued.contains { $0.markerId == markerId }
    }

    /// Shows `offer` if none is showing, otherwise queues it. A repeat of an
    /// id already held is ignored.
    mutating func add(_ offer: LibraryOffer) {
        guard !contains(markerId: offer.markerId) else { return }
        if pending == nil {
            pending = offer
        } else {
            queued.append(offer)
        }
    }

    /// Drops the offer with `markerId`, wherever it is. If it was the one on
    /// screen, the next in line takes its place.
    mutating func remove(markerId: String) {
        queued.removeAll { $0.markerId == markerId }
        if pending?.markerId == markerId {
            pending = queued.isEmpty ? nil : queued.removeFirst()
        }
    }

    /// Drops every offer `shouldDrop` returns true for.
    mutating func prune(where shouldDrop: (LibraryOffer) -> Bool) {
        let held = (pending.map { [$0] } ?? []) + queued
        for offer in held where shouldDrop(offer) {
            remove(markerId: offer.markerId)
        }
    }
}
