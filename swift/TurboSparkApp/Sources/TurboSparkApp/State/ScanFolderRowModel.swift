import Foundation
import TurboSpark

/// What one look at an additional scan folder found: whether it can be read and,
/// when it can, how many models are in it.
struct ScanFolderReading: Equatable, Sendable {
    let status: ScanFolderStatus
    /// Meaningful only for `.connected`. Zero otherwise, and never shown: a
    /// folder nobody looked inside has no count, which is not the same as none.
    let modelCount: Int
}

/// Everything one additional-scan-folder row says (requirement 3.9, design.md
/// "Modified Files": `CustomModelFoldersSectionView`).
///
/// **IT IS PURE.** Every input is passed in, so a table test pins each state's
/// words, glyph, dimming, Reveal and spoken label without a view. The row draws
/// these fields and nothing else.
///
/// **A FOLDER THAT CANNOT BE READ HAS NO COUNT.** An unplugged drive is not a
/// drive with zero models (requirement 3.9), so a `.notConnected` or `.notFound`
/// row says what is wrong, in a glyph AND a word (DESIGN.md section 4), and
/// never "Models: 0". Before the first reading lands (`reading == nil`) the row
/// says neither: it has not looked yet.
///
/// **THE TWO CAUSES HAVE TWO SHAPES.** A missing drive reads like a missing
/// library everywhere else in the app (the minus circle and "Not connected"); a
/// folder that is gone from a drive that IS there is a triangle and "Folder not
/// found". The user plugs in a drive for the first and fixes the list for the
/// second.
///
/// **THE VOLUME NAME COMES FROM A DRIVE.** It, and the path it is part of, go
/// through `UntrustedText.display` before they reach the screen or VoiceOver.
/// The raw `path` stays for Reveal and Remove, which act on the real folder.
struct ScanFolderRowModel: Equatable, Identifiable, Sendable {
    /// The folder as configured. Reveal and Remove act on this.
    let path: String
    /// The same, escaped and cut for display.
    let displayPath: String
    /// Nil until the first reading lands.
    let status: ScanFolderStatus?
    /// "Models: 3", only for a folder that was read and is connected.
    let countText: String?
    /// "Not connected" or "Folder not found". Nil when connected or not read yet.
    let statusText: String?
    let statusGlyph: String?
    /// "Connect T7 to scan this folder.", only for a drive that is away.
    let hintText: String?
    /// Secondary text for a folder that cannot be read.
    let isDimmed: Bool
    let reveal: LibraryActionAvailability
    /// Removing a folder from the list touches no file, so it is always on: a
    /// dead path is exactly when the user wants it.
    let remove: LibraryActionAvailability
    /// Path, then status or count, then the drive to connect: one element.
    let accessibilityLabel: String

    var id: String { path }

    /// The folder cannot be read now. The view reads such a row as one element.
    var isOffline: Bool { statusText != nil }

    init(path: String, reading: ScanFolderReading?, language: AppLanguage = AppStrings.currentLanguage) {
        let displayPath = UntrustedText.display(path, limit: ModelLibraryRowModel.pathLimit)

        var countText: String?
        var statusText: String?
        var statusGlyph: String?
        var hintText: String?
        var revealReason: String?
        switch reading?.status {
        case nil:
            break
        case .connected?:
            let count = reading?.modelCount ?? 0
            countText = AppStrings.localized("Models: \(count)", language: language)
        case .notConnected(let volumeName)?:
            let drive = UntrustedText.display(volumeName, limit: Self.volumeNameLimit)
            statusText = AppStrings.localized(
                String.LocalizationValue(stringLiteral: LibraryRowStatus.notConnected.title), language: language)
            statusGlyph = LibraryRowStatus.notConnected.glyph
            hintText = AppStrings.localized("Connect \(drive) to scan this folder.", language: language)
            revealReason = AppStrings.localized("Connect the drive to use this.", language: language)
        case .notFound?:
            statusText = AppStrings.localized("Folder not found", language: language)
            statusGlyph = LibraryRowGlyph.warning
            // Finder opens nothing for a folder that is not there, and a button
            // that silently does nothing is worse than one that says why.
            revealReason = statusText
        }

        self.path = path
        self.displayPath = displayPath
        self.status = reading?.status
        self.countText = countText
        self.statusText = statusText
        self.statusGlyph = statusGlyph
        self.hintText = hintText
        self.isDimmed = statusText != nil
        self.reveal = LibraryActionAvailability(isEnabled: revealReason == nil, disabledReason: revealReason)
        self.remove = LibraryActionAvailability(isEnabled: true, disabledReason: nil)

        // The order the row reads in, a sentence's full stop dropped so the
        // parts join into one.
        let parts = [displayPath, statusText ?? countText, hintText].compactMap { $0 }
        self.accessibilityLabel = parts
            .map { $0.hasSuffix(".") ? String($0.dropLast()) : $0 }
            .joined(separator: ". ")
    }

    /// A drive name is a label, not a sentence: this is generous and only bounds
    /// what a marker or a volume could make the row carry.
    static let volumeNameLimit = 60
}
