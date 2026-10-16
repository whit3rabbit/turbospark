import SwiftUI

/// Choosing a reference voice from the library: recently used references
/// from past clone takes first, then every saved recording, import and take
/// with audio, searchable and filterable to favorites.
@MainActor
struct VoiceReferencePickerSheet: View {
    @ObservedObject var controller: AudioWorkspaceController
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var favoritesOnly = false
    @FocusState private var searchFocused: Bool

    private var recents: [AudioRecentVoiceReference] {
        guard query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !favoritesOnly else { return [] }
        return controller.recentVoiceReferences
    }

    private var candidates: [AudioLibraryItem] {
        AudioWorkspaceController.voiceReferenceCandidates(in: controller.items, query: query, favoritesOnly: favoritesOnly)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Choose from library", bundle: .module)
                .themedFont(.title2, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            HStack(spacing: AppMetrics.actionSpacing) {
                AppSearchField(text: $query, prompt: Text("Search recordings and takes", bundle: .module),
                               maxWidth: .infinity, focus: $searchFocused)
                AppChip(Text("Favorites", bundle: .module), systemImage: favoritesOnly ? "star.fill" : "star", isOn: favoritesOnly) {
                    favoritesOnly.toggle()
                }
            }
            list
            HStack {
                Spacer()
                Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                    .appAction(.secondary)
                    .keyboardShortcut(.cancelAction)
            }
        }
        .padding(24)
        .frame(minWidth: 460, idealWidth: 560, minHeight: 420, idealHeight: 580)
        .themedFont(.base)
        .onAppear { searchFocused = true }
    }

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 6) {
                if !recents.isEmpty {
                    AppSectionHeader(title: Text("Recently used", bundle: .module)).padding(.bottom, 2)
                    ForEach(recents) { recent in
                        row(symbol: "clock.arrow.circlepath", title: recent.reference.title,
                            detail: Text(verbatim: audioTime(recent.reference.start) + " - " + audioTime(recent.reference.start + recent.reference.duration)),
                            date: recent.usedAt, duration: recent.reference.duration, favorite: false) {
                            controller.useVoiceReference(fromTake: recent.take)
                        }
                    }
                    AppSectionHeader(title: Text("Saved audio", bundle: .module)).padding(.top, 10).padding(.bottom, 2)
                }
                if candidates.isEmpty {
                    AppEmptyState(systemImage: "waveform", title: Text("No recordings or takes found", bundle: .module),
                                  message: Text("Use a clear recording of one person speaking for 1 to 30 seconds.", bundle: .module))
                } else {
                    ForEach(candidates) { item in
                        row(symbol: item.kind.voiceReferenceSymbol, title: item.title, detail: item.kind.voiceReferenceTitle,
                            date: item.createdAt, duration: item.duration, favorite: item.favorite) {
                            controller.setVoiceReference(item)
                        }
                    }
                }
            }
            .padding(.vertical, 2)
        }
        .disabled(!controller.canEditVoiceReference)
    }

    private func row(symbol: String, title: String, detail: Text, date: Date, duration: Double, favorite: Bool,
                     choose: @escaping () -> Void) -> some View {
        VoiceReferencePickerRow(symbol: symbol, title: title, detail: detail, date: date, duration: duration, favorite: favorite) {
            choose()
            dismiss()
        }
    }
}

@MainActor
private struct VoiceReferencePickerRow: View {
    let symbol: String
    let title: String
    let detail: Text
    let date: Date
    let duration: Double
    let favorite: Bool
    let action: () -> Void

    @State private var hovered = false
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: symbol)
                    .themedFont(.base).foregroundStyle(.appAccent)
                    .frame(width: 28, height: 28)
                    .background(.appAccent.opacity(0.12), in: Circle())
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(verbatim: title).lineLimit(1).truncationMode(.middle)
                    HStack(spacing: 10) {
                        detail
                        Text(date, format: .dateTime.month(.abbreviated).day().hour().minute())
                    }
                    .themedFont(.tiny).foregroundStyle(.appSecondary).lineLimit(1)
                }
                Spacer(minLength: 8)
                if favorite {
                    Image(systemName: "star.fill").themedFont(.small).foregroundStyle(.appAccent)
                        .accessibilityLabel(Text("Favorite", bundle: .module))
                }
                Text(verbatim: audioTime(duration)).monospacedDigit().foregroundStyle(.appSecondary)
            }
            .padding(.horizontal, 12).padding(.vertical, 9)
            .background(hovered && isEnabled ? AnyShapeStyle(.appElevated) : AnyShapeStyle(.appSurface),
                        in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.985))
        .onHover { hovered = $0 }
        .tsAnimation(TSMotion.hover, value: hovered)
        .help(Text(verbatim: title))
        .accessibilityElement(children: .combine)
    }
}

extension AudioLibraryItem.Kind {
    /// The glyph a reference source shows in pickers and the reference card.
    var voiceReferenceSymbol: String {
        switch self {
        case .recording: return "record.circle"
        case .imported: return "square.and.arrow.down"
        case .transcription: return "text.alignleft"
        case .voiceover: return "person.wave.2"
        case .music: return "music.note"
        case .enhancement: return "wand.and.stars"
        case .separation: return "square.split.2x1"
        case .experiment: return "flask"
        }
    }

    var voiceReferenceTitle: Text {
        switch self {
        case .recording: Text("Recording", bundle: .module)
        case .imported: Text("Imported", bundle: .module)
        case .transcription: Text("Transcribe", bundle: .module)
        case .voiceover: Text("Voiceover", bundle: .module)
        case .music: Text("Music", bundle: .module)
        case .enhancement: Text("Enhance speech", bundle: .module)
        case .separation: Text("Separate stems", bundle: .module)
        case .experiment: Text("Advanced", bundle: .module)
        }
    }
}
