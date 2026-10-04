import SwiftUI

enum ChromeBookmarkImportStage: Equatable {
    case chooseProfiles
    case preview
}

enum ChromeBookmarkImportFeedback: Equatable {
    case chooseProfile
    case chooseBookmark
    case savedBookmarkLimit
}

struct ChromeBookmarkPreviewRow: Equatable, Identifiable, Sendable {
    enum Kind: Equatable, Sendable {
        case folder
        case bookmark
    }

    let id: String
    let profileID: String
    let title: String
    let detail: String?
    let depth: Int
    let kind: Kind
}

@MainActor
final class ChromeBookmarkImportPreviewViewModel: ObservableObject {
    @Published private(set) var stage: ChromeBookmarkImportStage = .chooseProfiles
    @Published private(set) var enumeration: ChromeBookmarkProfileEnumeration?
    @Published private(set) var previews: [ChromeBookmarkProfilePreview] = []
    @Published private(set) var selectedProfileIDs = Set<String>()
    @Published private(set) var selectedIdentifiersByProfileID: [String: Set<String>] = [:]
    @Published private(set) var importReport: ChromeBookmarkImportResult?
    @Published private(set) var feedback: ChromeBookmarkImportFeedback?

    private let importer: ChromeBookmarkImporter
    private let existingBookmarks: [BrowserBookmarkTree]
    private let onSave: ([BrowserBookmarkTree]) -> Bool
    private var didCommitImport = false

    init(
        importer: ChromeBookmarkImporter,
        existingBookmarks: [BrowserBookmarkTree] = [],
        onSave: @escaping ([BrowserBookmarkTree]) -> Bool = { _ in true }
    ) {
        self.importer = importer
        self.existingBookmarks = existingBookmarks
        self.onSave = onSave
    }

    func scanProfiles() {
        enumeration = importer.enumerateProfiles()
        selectedProfileIDs.removeAll()
        previews.removeAll()
        selectedIdentifiersByProfileID.removeAll()
        importReport = nil
        feedback = nil
        stage = .chooseProfiles
    }

    func setProfileSelected(_ profileID: String, selected: Bool) {
        guard enumeration?.profiles.contains(where: { $0.id == profileID }) == true else { return }
        if selected {
            selectedProfileIDs.insert(profileID)
        } else {
            selectedProfileIDs.remove(profileID)
        }
        feedback = nil
    }

    @discardableResult
    func previewSelectedProfiles() -> Bool {
        guard !selectedProfileIDs.isEmpty else {
            feedback = .chooseProfile
            return false
        }
        previews = importer.preview(profileIDs: selectedProfileIDs)
        selectedIdentifiersByProfileID = Dictionary(
            uniqueKeysWithValues: previews.map { ($0.profile.id, Set<String>()) }
        )
        stage = .preview
        feedback = nil
        return true
    }

    func returnToProfileSelection() {
        previews.removeAll()
        selectedIdentifiersByProfileID.removeAll()
        importReport = nil
        stage = .chooseProfiles
        feedback = nil
    }

    func rows(for profileID: String) -> [ChromeBookmarkPreviewRow] {
        guard let preview = previews.first(where: { $0.profile.id == profileID }), preview.issue == nil else {
            return []
        }
        var result: [ChromeBookmarkPreviewRow] = []
        for folder in preview.folders {
            appendRows(for: folder, profileID: profileID, depth: 0, into: &result)
        }
        return result
    }

    func isSelected(_ row: ChromeBookmarkPreviewRow) -> Bool {
        let selected = selectedIdentifiersByProfileID[row.profileID] ?? []
        let affected = identifiers(for: row)
        return !affected.isEmpty && affected.isSubset(of: selected)
    }

    func toggle(_ row: ChromeBookmarkPreviewRow) {
        guard let preview = previews.first(where: { $0.profile.id == row.profileID }),
              preview.selectableIdentifiers.contains(row.id)
        else {
            return
        }
        let affected = identifiers(for: row)
        var selected = selectedIdentifiersByProfileID[row.profileID] ?? []
        if affected.isSubset(of: selected) {
            selected.subtract(affected)
            // Parent IDs select whole subtrees at import time, so narrow their grants too.
            for folder in preview.folders {
                _ = removeAncestorSelections(of: affected, in: folder, from: &selected)
            }
        } else {
            selected.formUnion(affected)
        }
        selectedIdentifiersByProfileID[row.profileID] = selected
        feedback = nil
    }

    @discardableResult
    func confirmImport() -> Bool {
        guard !didCommitImport else { return true }
        let report = importer.importConfirmed(
            selectionsByProfileID: selectedIdentifiersByProfileID
        )
        guard !report.trees.isEmpty else {
            feedback = .chooseBookmark
            importReport = report
            return false
        }
        guard onSave(existingBookmarks + report.trees) else {
            feedback = .savedBookmarkLimit
            importReport = report
            return false
        }
        didCommitImport = true
        feedback = nil
        importReport = report
        return true
    }

    func cancel() {
        selectedProfileIDs.removeAll()
        selectedIdentifiersByProfileID.removeAll()
        previews.removeAll()
        importReport = nil
        feedback = nil
        stage = .chooseProfiles
    }

    private func appendRows(
        for folder: BrowserBookmarkFolder,
        profileID: String,
        depth: Int,
        into rows: inout [ChromeBookmarkPreviewRow]
    ) {
        rows.append(ChromeBookmarkPreviewRow(
            id: folder.id,
            profileID: profileID,
            title: folder.name,
            detail: nil,
            depth: depth,
            kind: .folder
        ))
        for bookmark in folder.bookmarks {
            let origin = URL(string: bookmark.url).flatMap(BrowserOrigin.init(url:))?.canonicalString
            rows.append(ChromeBookmarkPreviewRow(
                id: bookmark.id,
                profileID: profileID,
                title: bookmark.title,
                detail: origin,
                depth: depth + 1,
                kind: .bookmark
            ))
        }
        for child in folder.folders {
            appendRows(for: child, profileID: profileID, depth: depth + 1, into: &rows)
        }
    }

    private func identifiers(for row: ChromeBookmarkPreviewRow) -> Set<String> {
        guard row.kind == .folder,
              let root = previews.first(where: { $0.profile.id == row.profileID })?.folders
        else {
            return [row.id]
        }
        guard let folder = findFolder(row.id, in: root) else { return [row.id] }
        var result: Set<String> = []
        appendIdentifiers(for: folder, to: &result)
        return result
    }

    private func findFolder(_ identifier: String, in folders: [BrowserBookmarkFolder]) -> BrowserBookmarkFolder? {
        for folder in folders {
            if folder.id == identifier { return folder }
            if let nested = findFolder(identifier, in: folder.folders) { return nested }
        }
        return nil
    }

    private func appendIdentifiers(for folder: BrowserBookmarkFolder, to result: inout Set<String>) {
        result.insert(folder.id)
        result.formUnion(folder.bookmarks.map(\.id))
        for nested in folder.folders {
            appendIdentifiers(for: nested, to: &result)
        }
    }

    private func removeAncestorSelections(
        of deselected: Set<String>,
        in folder: BrowserBookmarkFolder,
        from selected: inout Set<String>
    ) -> Bool {
        var containsDeselection = deselected.contains(folder.id)
            || folder.bookmarks.contains { deselected.contains($0.id) }
        for child in folder.folders {
            if removeAncestorSelections(of: deselected, in: child, from: &selected) {
                containsDeselection = true
            }
        }
        if containsDeselection { selected.remove(folder.id) }
        return containsDeselection
    }
}

@MainActor
struct ChromeBookmarkImportPreviewView: View {
    @StateObject private var viewModel: ChromeBookmarkImportPreviewViewModel
    @Environment(\.dismiss) private var dismiss

    init(viewModel: ChromeBookmarkImportPreviewViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    init(model: AppModel) {
        let viewModel = ChromeBookmarkImportPreviewViewModel(
            importer: ChromeBookmarkImporter(),
            existingBookmarks: model.browserSettings.bookmarks
        ) { bookmarks in
            var settings = model.browserSettings
            guard settings.replaceBookmarks(bookmarks) else { return false }
            model.browserSettings = settings
            model.persistSettingsDebounced()
            return true
        }
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Import Chrome bookmarks", bundle: .module)
                    .themedFont(.title2, weight: .semibold)
                Spacer()
                Button {
                    viewModel.cancel()
                    dismiss()
                } label: {
                    Text("Cancel", bundle: .module)
                }
                .keyboardShortcut(.cancelAction)
            }

            if let enumeration = viewModel.enumeration {
                switch viewModel.stage {
                case .chooseProfiles:
                    profileSelection(enumeration)
                case .preview:
                    bookmarkPreview(enumeration)
                }
            } else {
                ProgressView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            if let feedback = viewModel.feedback {
                Text(feedbackKey(feedback), bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.red)
                    .accessibilityIdentifier("chrome-bookmark-import-feedback")
            }
        }
        .padding(24)
        .frame(minWidth: 520, minHeight: 420)
        .task {
            if viewModel.enumeration == nil { viewModel.scanProfiles() }
        }
    }

    private func profileSelection(_ enumeration: ChromeBookmarkProfileEnumeration) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Choose profiles to preview", bundle: .module)
                .themedFont(.small, weight: .semibold)

            if enumeration.profiles.isEmpty {
                Text("No Chrome profiles with bookmarks found.", bundle: .module)
                    .foregroundStyle(.appSecondary)
                ForEach(Array(enumeration.issues.enumerated()), id: \.offset) { _, issue in
                    Text(issueKey(issue), bundle: .module)
                        .foregroundStyle(.appSecondary)
                }
            } else {
                List(enumeration.profiles) { profile in
                    Toggle(
                        profile.displayLabel,
                        isOn: Binding(
                            get: { viewModel.selectedProfileIDs.contains(profile.id) },
                            set: { viewModel.setProfileSelected(profile.id, selected: $0) }
                        )
                    )
                }
                .listStyle(.inset)
            }

            HStack {
                Spacer()
                Button {
                    _ = viewModel.previewSelectedProfiles()
                } label: {
                    Text("Preview", bundle: .module)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(enumeration.profiles.isEmpty)
            }
        }
    }

    private func bookmarkPreview(_ enumeration: ChromeBookmarkProfileEnumeration) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Button {
                    viewModel.returnToProfileSelection()
                } label: {
                    Text("Back", bundle: .module)
                }
                Text("Preview", bundle: .module)
                    .themedFont(.small, weight: .semibold)
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    ForEach(viewModel.previews, id: \.profile.id) { preview in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(preview.profile.displayLabel)
                                .themedFont(.base, weight: .semibold)
                            if let issue = preview.issue {
                                Text(issueKey(issue), bundle: .module)
                                    .foregroundStyle(.red)
                            } else {
                                ForEach(viewModel.rows(for: preview.profile.id)) { row in
                                    selectionRow(row)
                                }
                                if preview.skippedItemCount > 0 {
                                    countLabel("Skipped items", count: preview.skippedItemCount)
                                        .foregroundStyle(.appSecondary)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    ForEach(Array(enumeration.issues.enumerated()), id: \.offset) { _, issue in
                        Text(issueKey(issue), bundle: .module)
                            .foregroundStyle(.appSecondary)
                    }
                    if let report = viewModel.importReport {
                        VStack(alignment: .leading, spacing: 4) {
                            countLabel("Imported", count: report.importedBookmarkCount)
                            countLabel("Skipped items", count: report.skippedItemCount)
                            ForEach(Array(report.issues.enumerated()), id: \.offset) { _, issue in
                                Text(issueKey(issue), bundle: .module)
                                    .foregroundStyle(.appSecondary)
                            }
                        }
                        .accessibilityIdentifier("chrome-bookmark-import-report")
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack {
                Spacer()
                Button {
                    if viewModel.confirmImport() { dismiss() }
                } label: {
                    Text("Import selected bookmarks", bundle: .module)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(viewModel.previews.allSatisfy { $0.issue != nil })
            }
        }
    }

    private func selectionRow(_ row: ChromeBookmarkPreviewRow) -> some View {
        Toggle(
            isOn: Binding(
                get: { viewModel.isSelected(row) },
                set: { _ in viewModel.toggle(row) }
            )
        ) {
            HStack(spacing: 6) {
                Image(systemName: row.kind == .folder ? "folder" : "bookmark")
                    .foregroundStyle(.appSecondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(row.title)
                        .lineLimit(1)
                    if let detail = row.detail {
                        Text(detail)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(1)
                    }
                }
            }
        }
        .toggleStyle(.checkbox)
        .padding(.leading, CGFloat(row.depth) * 16)
        .accessibilityIdentifier("chrome-bookmark-row-\(row.id)")
    }

    private func countLabel(_ key: LocalizedStringKey, count: Int) -> some View {
        HStack(spacing: 4) {
            Text(key, bundle: .module)
            Text(count, format: .number)
        }
        .themedFont(.small)
    }

    private func issueKey(_ issue: ChromeBookmarkImportIssue) -> LocalizedStringKey {
        switch issue {
        case .profileRootUnavailable, .noProfilesFound:
            "No Chrome profiles with bookmarks found."
        case .bookmarksFileMissing:
            "The selected profile no longer has a Bookmarks file."
        case .bookmarksFileUnreadable:
            "The selected profile Bookmarks file could not be read."
        case .bookmarksFileTooLarge:
            "The selected profile Bookmarks file exceeds the size limit."
        case .malformedBookmarks:
            "The selected profile Bookmarks file could not be parsed."
        case .profileUnavailable:
            "The selected Chrome profile is no longer available."
        case .unknownSelection:
            "Some selected bookmark items are unavailable."
        }
    }

    private func feedbackKey(_ feedback: ChromeBookmarkImportFeedback) -> LocalizedStringKey {
        switch feedback {
        case .chooseProfile, .chooseBookmark: "Select at least one item to continue."
        case .savedBookmarkLimit: "Saved bookmarks exceed the 2 MiB settings limit."
        }
    }
}
