import AppKit
import SwiftUI
import TurboSpark

/// The Model libraries section of Models and Storage: This Mac, every library
/// the user added, and a not-connected placeholder for a store Move Store moved
/// to a drive that is away (tasks 6.1 and 6.2; requirements 1.3 to 1.10, 3.8,
/// 5.1, 5.2, 5.6 and 9.3).
///
/// **IT OBSERVES `model.libraryState` DIRECTLY.** `AppModel` does not publish it
/// (a mount would redraw every screen), so this view holds the state as its own
/// `@ObservedObject` and redraws on a mount, an unmount or a refresh.
///
/// **IT NEVER RAISES AN ERROR.** A library that is away is a row that says so
/// (requirement 3.4). A refusal of something the user just did (a folder on a
/// network share, a read-only drive) is a message in this section, with a glyph
/// and a sentence, until it is dismissed or the next action replaces it. The
/// app error and the toast are not used (`ModelLibrariesController`).
///
/// Isolated explicitly: its helpers call each other, and only `body` is isolated
/// by the protocol on the macOS 14 SDK.
@MainActor
struct ModelLibrariesSectionView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var libraryState: LibraryState
    @StateObject private var controller: ModelLibrariesController

    /// Read so the strings built below follow the language picker: they come
    /// from `AppStrings`, which has no dependency SwiftUI can see.
    @Environment(\.locale) private var locale

    @State private var removal: RemovalCandidate?
    @State private var confirmingAbandon = false

    /// The library the destructive confirmation is about.
    struct RemovalCandidate: Identifiable, Equatable {
        let id: String
        let title: String
        let isDefault: Bool
    }

    /// - Parameter controller: a test passes its own, wired to a fake speed
    ///   test and a recording announcer. The app lets the view build the real
    ///   one.
    init(model: AppModel, controller: ModelLibrariesController? = nil) {
        self.model = model
        _libraryState = ObservedObject(wrappedValue: model.libraryState)
        _controller = StateObject(
            wrappedValue: controller
                ?? ModelLibrariesController(
                    libraryState: model.libraryState, model: model,
                    migration: { [weak model] in model?.moveStoreMigration }))
    }

    private var language: AppLanguage {
        _ = locale
        return AppStrings.currentLanguage
    }

    private var rows: [ModelLibraryRowModel] {
        ModelLibraryRowModel.rows(from: libraryState, language: language)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            VStack(alignment: .leading, spacing: 10) {
                // A library found on a drive that just connected (task 6.6): the
                // same banner as the Installed Models manager, from the same
                // state, so answering in one removes it from the other.
                if libraryState.pendingOffer != nil {
                    LibraryOfferBanner(model: model, libraryState: libraryState)
                }
                Text("Keep models on drives connected to this Mac. Your settings and model list stay on this Mac when a drive is unplugged.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let message = controller.message {
                    messageBanner(message)
                }
                ForEach(rows) { row in
                    ModelLibraryRowView(
                        row: row,
                        changesBlocked: controller.isChangeBlocked,
                        speedTest: controller.speedTests[row.id],
                        perform: { perform($0, on: row) },
                        cancelSpeedTest: { controller.cancelSpeedTest(libraryId: row.id) },
                        dismissSpeedTestFailure: { controller.dismissSpeedTestFailure(libraryId: row.id) })
                }
                if let migration = model.moveStoreMigration, libraryState.pendingMovedStore == nil {
                    MovedStoreNoticeView(migration: migration) { confirmingAbandon = true }
                }
            }
            .modelsSettingsCard()
        }
        // The counts and free space of a section opened after an install are
        // read again, off the main actor, rather than shown from the last mount.
        .task { await libraryState.refresh() }
        // A test left running must not keep reading a drive nobody is looking at.
        .onDisappear { controller.cancelAllSpeedTests() }
        .confirmationDialog(
            Text(verbatim: removalTitle),
            isPresented: Binding(get: { removal != nil }, set: { if !$0 { removal = nil } }),
            titleVisibility: .visible,
            presenting: removal
        ) { candidate in
            Button(role: .destructive) {
                Task { await controller.remove(id: candidate.id) }
            } label: {
                Text("Remove Library", bundle: .module)
            }
            Button(role: .cancel) {
            } label: {
                Text("Cancel", bundle: .module)
            }
        } message: { candidate in
            // The files stay on the drive (requirement 1.8); the default
            // library hands the default back to This Mac.
            if candidate.isDefault {
                Text("TurboSpark stops listing this library and its models, and new downloads go to This Mac. The files on the drive are not changed.", bundle: .module)
            } else {
                Text("TurboSpark stops listing this library and its models. The files on the drive are not changed.", bundle: .module)
            }
        }
        .confirmationDialog(
            Text("Stop waiting for this drive?", bundle: .module),
            isPresented: $confirmingAbandon,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                controller.abandonConversion()
            } label: {
                Text("Stop waiting for this drive", bundle: .module)
            }
            Button(role: .cancel) {
            } label: {
                Text("Cancel", bundle: .module)
            }
        } message: {
            Text("TurboSpark forgets the earlier model store. Its models stay on the drive and are not changed. To use them later, add the folder as a library.", bundle: .module)
        }
    }

    // MARK: - Header

    private var header: some View {
        AppAdaptiveHeader {
            Label { Text("Model libraries", bundle: .module) } icon: { Image(systemName: "externaldrive") }
                .settingsControl("Model libraries", pane: .models, timing: .action)
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
        } actions: {
            Button {
                chooseFolder()
            } label: {
                Label {
                    Text("Add Library\u{2026}", bundle: .module)
                } icon: {
                    Image(systemName: "plus")
                }
            }
            .appAction(.secondary, size: .small)
            .disabled(controller.isChangeBlocked)
            .help(
                controller.isChangeBlocked
                    ? Text("Not available while a model is installing or moving.", bundle: .module)
                    : Text("Add Library\u{2026}", bundle: .module))
            .settingsControl("Add Library\u{2026}", pane: .models, timing: .action)
        }
    }

    // MARK: - Message

    private func messageBanner(_ message: LibraryMessage) -> some View {
        HStack(alignment: .top, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                if message.kind == .error {
                    Image(systemName: LibraryRowGlyph.warning).foregroundStyle(.red).accessibilityHidden(true)
                } else {
                    Image(systemName: LibraryRowGlyph.info).foregroundStyle(.green).accessibilityHidden(true)
                }
                Text(verbatim: message.text)
                    .themedFont(.small)
                    .foregroundStyle(.appText)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .accessibilityElement(children: .combine)
            Spacer(minLength: 8)
            AppIconButton("xmark", label: Text("Dismiss", bundle: .module)) {
                controller.dismissMessage()
            }
            .appAction(.quiet, size: .small)
        }
        .padding(10)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(message.kind == .error ? Color.red.opacity(0.55) : Color.green.opacity(0.55), lineWidth: 1)
        )
        .accessibilityAction(named: Text("Dismiss", bundle: .module)) {
            controller.dismissMessage()
        }
    }

    // MARK: - Actions

    private func perform(_ action: LibraryRowAction, on row: ModelLibraryRowModel) {
        switch action {
        case .setDefault:
            Task { await controller.setDefault(id: row.id) }
        case .testSpeed:
            controller.testSpeed(libraryId: row.id)
        case .reveal:
            controller.revealInFinder(libraryId: row.id)
        case .remove:
            removal = RemovalCandidate(id: row.id, title: row.title, isDefault: row.isDefault)
        case .abandonConversion:
            confirmingAbandon = true
        }
    }

    private var removalTitle: String {
        guard let removal else { return "" }
        return AppStrings.localized("Remove the library \(removal.title)?", language: language)
    }

    /// Opens the panel at the volumes folder, where an external drive is, and
    /// adds what the user picks. The engine decides whether it may be a library.
    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: "/Volumes", isDirectory: true)
        panel.prompt = AppStrings.localized("Add Library")
        panel.message = AppStrings.localized("Choose a drive or a folder on a drive to use as a model library")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await controller.add(path: url.path) }
    }
}

/// A conversion of a store Move Store moved that has not finished and has no
/// placeholder row: the drive answered but something failed (a refusal, a copy
/// that could not be made). It says so in one generic sentence, never in the
/// engine's, and offers to stop (the 5.10 note on permanent stuck states).
///
/// It observes the migration itself because `AppModel` publishes neither the
/// migration nor its state.
private struct MovedStoreNoticeView: View {
    @ObservedObject var migration: MoveStoreMigration
    let stop: () -> Void

    var body: some View {
        if migration.isPending, !migration.lastCopyFailures.isEmpty || migration.lastFailure != nil {
            HStack(alignment: .top, spacing: 8) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Image(systemName: LibraryRowGlyph.warning)
                            .foregroundStyle(.orange)
                            .accessibilityHidden(true)
                        sentence
                            .themedFont(.small)
                            .foregroundStyle(.appText)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Button {
                        stop()
                    } label: {
                        Label {
                            Text(LocalizedStringKey(LibraryRowAction.abandonConversion.title), bundle: .module)
                        } icon: {
                            Image(systemName: LibraryRowAction.abandonConversion.systemImage)
                        }
                    }
                    .appAction(.destructive, size: .small)
                }
                Spacer(minLength: 0)
            }
            .padding(10)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .accessibilityElement(children: .contain)
        }
    }

    @ViewBuilder private var sentence: some View {
        if !migration.lastCopyFailures.isEmpty {
            Text("Some models from your earlier model store could not be copied to this Mac yet. TurboSpark tries again each time it starts and when the drive connects.", bundle: .module)
        } else {
            Text("TurboSpark has not finished converting your earlier model store. It tries again each time it starts and when the drive connects.", bundle: .module)
        }
    }
}
