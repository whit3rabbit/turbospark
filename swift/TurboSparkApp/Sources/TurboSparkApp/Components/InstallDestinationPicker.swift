import SwiftUI
import TurboSpark

/// The menu that says which library a model is installed into (task 6.9,
/// requirements 2.1 and 2.2), with each choice's free space.
///
/// **IT IS DRAWN ONLY WHEN THERE IS A CHOICE.** With This Mac alone connected
/// there is no menu and the install goes there, as it always did. Embedding and
/// OCR installs never show it (requirement 2.5): their surfaces do not embed it,
/// and their downloads carry no library.
///
/// **IT PRESELECTS THE DEFAULT.** An untouched menu has no selection (`nil`) and
/// follows `LibraryState.destinationDefault`. When the default library is away the
/// menu preselects This Mac and says why, in a glyph and a sentence; that note is
/// drawn even when the menu is not, because the install still goes to This Mac and
/// the user who chose a drive for new downloads is owed the reason.
///
/// **THE CHOICE IS PER INSTALL.** It is the caller's `@State`, not remembered
/// across launches, and each surface starts from the default again.
///
/// **IT DRAWS AN `InstallDestinationModel` AND NOTHING ELSE,** so a table test pins
/// every row and every word. Library names are from a drive: the model escaped and
/// cut them, and this only draws them (`Text(verbatim:)`, never a key).
///
/// It observes `LibraryState` directly: `AppModel` does not publish it, so a mount
/// would not redraw this menu otherwise. The move sheet (task 6.11) takes the same
/// menu.
@MainActor
struct InstallDestinationPicker: View {
    let model: AppModel
    @ObservedObject private var libraryState: LibraryState
    /// The user's choice: a library id, or nil to follow the default.
    @Binding var selection: String?

    /// Read so the strings built from `AppStrings` follow the language picker.
    @Environment(\.locale) private var locale

    init(model: AppModel, selection: Binding<String?>) {
        self.model = model
        _libraryState = ObservedObject(wrappedValue: model.libraryState)
        _selection = selection
    }

    private var language: AppLanguage {
        _ = locale
        return AppStrings.currentLanguage
    }

    var body: some View {
        let destinations = InstallDestinationModel(
            state: libraryState, thisMacFreeBytes: model.thisMacFreeSpaceReader(), language: language)
        if destinations.isShown || destinations.fallbackNote != nil {
            VStack(alignment: .leading, spacing: 6) {
                if destinations.isShown { menuRow(destinations) }
                if let note = destinations.fallbackNote { noteLine(note) }
            }
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("install-destination-picker")
        }
    }

    // MARK: - The menu

    /// The label and the menu side by side while they fit, one under the other
    /// when they do not (a long name, a long translation, the largest text size).
    /// An `HStack` mirrors in a right-to-left language by itself.
    private func menuRow(_ destinations: InstallDestinationModel) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                caption
                menu(destinations)
            }
            VStack(alignment: .leading, spacing: 4) {
                caption
                menu(destinations)
            }
        }
    }

    private var caption: some View {
        Text("Install to", bundle: .module)
            .themedFont(.small, weight: .medium)
            .foregroundStyle(.appSecondary)
            // The menu below carries the same name for VoiceOver.
            .accessibilityHidden(true)
    }

    private func menu(_ destinations: InstallDestinationModel) -> some View {
        Picker(selection: chosen(destinations)) {
            ForEach(destinations.menuChoices(for: selection)) { choice in
                Label {
                    Text(verbatim: choice.menuTitle)
                } icon: {
                    Image(systemName: Self.glyph(for: choice))
                }
                .tag(choice.id)
            }
        } label: {
            Text("Install to", bundle: .module)
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .fixedSize()
        .themedFont(.small)
        .help(Text("Choose the library this model is installed into.", bundle: .module))
        .accessibilityLabel(Text("Install to", bundle: .module))
    }

    /// The row shown as chosen: the user's pick (even a drive that has since left,
    /// which stays in the menu marked Not connected), or the preselected default.
    private func chosen(_ destinations: InstallDestinationModel) -> Binding<String> {
        Binding(
            get: { InstallDestinationModel.normalized(selection) ?? destinations.defaultId },
            set: { selection = $0 })
    }

    /// A shape for each kind of row, so the menu is never color alone.
    static func glyph(for choice: InstallDestinationChoice) -> String {
        if !choice.isConnected { return LibraryRowStatus.notConnected.glyph }
        return choice.isThisMac ? LibraryRowGlyph.thisMac : LibraryRowGlyph.drive
    }

    // MARK: - The note

    /// A glyph and a sentence, so the line survives color blindness and
    /// Differentiate Without Color.
    private func noteLine(_ sentence: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: "info.circle").foregroundStyle(AppThemeColor.appSecondary).accessibilityHidden(true)
            Text(verbatim: sentence)
                .themedFont(.small)
                .foregroundStyle(.appText)
        }
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
    }
}
