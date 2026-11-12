import AppKit
import SwiftUI

/// Builds a MiniMax Music style description from choices (genre, mood, tempo,
/// instruments, vocals, structure, production) and carries a short lyrics
/// guide. The generated text only reaches the recipe when the writer presses
/// "Use as Description", so browsing options never edits their work.
@MainActor
struct MusicPromptGeneratorView: View {
    @ObservedObject var controller: AudioWorkspaceController
    @Binding var selection: MusicPromptSelection
    @Environment(\.dismiss) private var dismiss
    @Environment(\.appTheme) private var theme
    @State private var tab = Tab.style

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    private enum Tab { case style, lyrics }

    private var caption: String { MusicPromptBuilder.caption(for: selection) }
    private var supportsLyrics: Bool { controller.selectedProfile?.capabilities.supportsLyrics == true }
    private var currentDescriptionIsEmpty: Bool {
        controller.recipe.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Lyric-related warnings only make sense when the model takes lyrics at
    /// all; otherwise the lyrics field is not even on screen.
    private var warnings: [MusicPromptWarning] {
        let lyricsAreInstrumental = AudioWorkspaceController.effectiveLyrics(controller.recipe.lyrics)
            == AudioWorkspaceController.instrumentalLyrics
        return MusicPromptBuilder.warnings(for: selection, lyricsAreInstrumental: lyricsAreInstrumental)
            .filter { supportsLyrics || ($0 != .vocalsWithoutLyrics && $0 != .instrumentalWithLyrics) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Prompt generator", bundle: .module).themedFont(.title2, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                if supportsLyrics {
                    Picker(selection: $tab) {
                        Text("Style", bundle: .module).tag(Tab.style)
                        Text("Lyrics guide", bundle: .module).tag(Tab.lyrics)
                    } label: { Text("Prompt generator", bundle: .module) }
                        .pickerStyle(.segmented).labelsHidden().fixedSize()
                }
            }
            ScrollView {
                Group {
                    if tab == .style { styleOptions } else { lyricsGuide }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.trailing, 12)
            }
            if tab == .style { preview }
            footer
        }
        .padding(24)
        .frame(
            minWidth: (540 * scale).rounded(),
            idealWidth: (640 * scale).rounded(),
            maxWidth: (760 * scale).rounded(),
            minHeight: (600 * scale).rounded(),
            idealHeight: (720 * scale).rounded(),
            maxHeight: (880 * scale).rounded()
        )
        .themedFont(.base)
        .background(.appPage)
    }

    // MARK: - Style

    private var styleOptions: some View {
        VStack(alignment: .leading, spacing: 20) {
            group("Style") {
                HStack(spacing: 12) {
                    genrePicker("Genre", $selection.genre)
                    genrePicker("Influence", $selection.influence).disabled(selection.genre == nil)
                }
                TextField(
                    text: $selection.scene,
                    prompt: Text("a late-night city drive", bundle: .module)
                ) { Text("Scene or purpose", bundle: .module) }
                    .textFieldStyle(.roundedBorder)
            }
            group("Mood", count: "\(selection.moods.count)/\(MusicPromptSelection.maxMoods)") {
                ForEach(MusicPromptCatalog.moods) { family in
                    multiChips(family.items, selected: selection.moods, limit: MusicPromptSelection.maxMoods) {
                        selection.moods = $0
                    }
                }
            }
            group("Tempo") {
                singleChips(MusicTempo.allCases.map(\.phrase), selected: selection.tempo?.phrase) { phrase in
                    selection.tempo = MusicTempo.allCases.first { $0.phrase == phrase }
                }
                HStack(spacing: 8) {
                    TextField(value: $selection.bpm, format: IntegerFormatStyle<Int>()) { Text(verbatim: "BPM") }
                        .textFieldStyle(.roundedBorder)
                        .frame(
                            minWidth: (70 * scale).rounded(),
                            idealWidth: (90 * scale).rounded(),
                            maxWidth: (120 * scale).rounded()
                        )
                    Text(verbatim: "\(MusicPromptSelection.bpmRange.lowerBound)-\(MusicPromptSelection.bpmRange.upperBound)")
                        .themedFont(.small).foregroundStyle(.appSecondary)
                }
            }
            group("Groove") {
                singleChips(MusicPromptCatalog.grooves, selected: selection.groove) { selection.groove = $0 }
            }
            group("Vocals") {
                Picker(selection: $selection.vocalMode) {
                    Text("Vocals", bundle: .module).tag(MusicVocalMode.vocal)
                    Text("Instrumental", bundle: .module).tag(MusicVocalMode.instrumental)
                } label: { Text("Vocals", bundle: .module) }
                    .pickerStyle(.segmented).labelsHidden().fixedSize()
                if selection.vocalMode == .vocal { vocalOptions }
            }
            group("Instruments", count: "\(selection.instruments.count)/\(MusicPromptSelection.maxInstruments)") {
                ForEach(MusicPromptCatalog.instruments) { family in
                    multiChips(family.items, selected: selection.instruments, limit: MusicPromptSelection.maxInstruments) {
                        selection.instruments = $0
                    }
                }
            }
            group("Structure") {
                Picker(selection: $selection.arc) {
                    Text("None", bundle: .module).tag(MusicArc?.none)
                    ForEach(MusicArc.allCases) { Text(verbatim: $0.sentence).tag(MusicArc?.some($0)) }
                } label: { Text("Structure", bundle: .module) }
                    .pickerStyle(.menu).labelsHidden()
            }
            group("Production", count: "\(selection.production.count)/\(MusicPromptSelection.maxProduction)") {
                multiChips(MusicPromptCatalog.production, selected: selection.production, limit: MusicPromptSelection.maxProduction) {
                    selection.production = $0
                }
            }
        }
    }

    private var vocalOptions: some View {
        VStack(alignment: .leading, spacing: 10) {
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                GridRow {
                    optionPicker("Voice", MusicPromptCatalog.voices, $selection.voice)
                    optionPicker("Tone", MusicPromptCatalog.tones, $selection.tone)
                }
                GridRow {
                    optionPicker("Delivery", MusicPromptCatalog.deliveries, $selection.delivery)
                    optionPicker("Emotion", MusicPromptCatalog.emotions, $selection.emotion)
                }
            }
            HStack {
                Text("Technique", bundle: .module).themedFont(.small, weight: .medium)
                Text(verbatim: "\(selection.techniques.count)/\(MusicPromptSelection.maxTechniques)")
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
            multiChips(MusicPromptCatalog.techniques, selected: selection.techniques, limit: MusicPromptSelection.maxTechniques) {
                selection.techniques = $0
            }
        }
    }

    private var preview: some View {
        VStack(alignment: .leading, spacing: 8) {
            ScrollView {
                Group {
                    if caption.isEmpty {
                        Text("Choose options to build a description.", bundle: .module).foregroundStyle(.appSecondary)
                    } else {
                        Text(verbatim: caption).textSelection(.enabled)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: (84 * scale).rounded()).padding(10)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous).stroke(.appBorder, lineWidth: 1) }
            ForEach(warnings, id: \.self) { warning in
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "exclamationmark.triangle").accessibilityHidden(true)
                    Text(Self.message(for: warning), bundle: .module)
                }
                .themedFont(.small).foregroundStyle(.appSecondary)
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 12) {
            if tab == .style {
                Button { selection = MusicPromptSelection() } label: { Text("Clear", bundle: .module) }
                    .appAction(.quiet)
                    .disabled(selection == MusicPromptSelection())
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(caption, forType: .string)
                } label: { Text("Copy", bundle: .module) }
                    .appAction(.secondary)
                    .disabled(caption.isEmpty)
            }
            Spacer()
            if tab == .style && !caption.isEmpty && !currentDescriptionIsEmpty {
                Text("Replaces the current description.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
            Button { dismiss() } label: { Text("Close", bundle: .module) }
                .appAction(.secondary)
                .keyboardShortcut(.cancelAction)
            if tab == .style {
                Button {
                    controller.recipe.caption = caption
                    dismiss()
                } label: { Text("Use as Description", bundle: .module) }
                    .appAction(.primary)
                    .keyboardShortcut(.defaultAction)
                    .disabled(caption.isEmpty || controller.isBusy)
            }
        }
    }

    // MARK: - Lyrics guide

    private var lyricsGuide: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Add a tag or a layout to the end of your lyrics.", bundle: .module).foregroundStyle(.appSecondary)
            group("Section tags") {
                FlowLayout(spacing: 6, lineSpacing: 6) {
                    ForEach(MusicLyricsGuide.sectionTags, id: \.self) { tag in
                        chip(MusicLyricsGuide.line(for: tag), selected: false, disabled: controller.isBusy) { add([tag]) }
                    }
                }
            }
            group("Song layouts") {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(MusicLyricsGuide.layouts, id: \.self) { layout in
                        Button { add(layout) } label: {
                            Text(verbatim: layout.joined(separator: ", "))
                                .themedFont(.small).multilineTextAlignment(.leading)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                                .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                                .overlay { RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous).stroke(.appBorder, lineWidth: 1) }
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain).disabled(controller.isBusy)
                    }
                }
            }
            group("Writing tips") {
                VStack(alignment: .leading, spacing: 8) {
                    tip("Put each section tag on its own line. Words on the same line as a tag are dropped.")
                    tip("Write for singing: short lines of similar length, in simple words.")
                    tip("Spell abbreviations and unusual names the way they should sound.")
                    tip("Lyrics set the song length. Short lyrics make a short song, and the duration is only an upper bound.")
                    tip("Keep lyrics out of the description. Describe how sections should sound there, such as a bigger chorus.")
                    tip("Repeat a hook or phrase across sections so the song feels connected.")
                }
            }
        }
    }

    private func add(_ tags: [String]) {
        controller.recipe.lyrics = MusicLyricsGuide.appending(tags, to: controller.recipe.lyrics)
    }

    private func tip(_ text: LocalizedStringKey) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "music.note").foregroundStyle(.appAccent).accessibilityHidden(true)
            Text(text, bundle: .module).fixedSize(horizontal: false, vertical: true)
        }
        .themedFont(.small)
    }

    private static func message(for warning: MusicPromptWarning) -> LocalizedStringKey {
        switch warning {
        case .noVoice: return "Name a voice type. Leaving it open often turns the track instrumental."
        case .vocalsWithoutLyrics: return "Vocals are on, but the lyrics are empty, so the track will render instrumental."
        case .instrumentalWithLyrics: return "Instrumental is on, but the lyrics contain words."
        case .instrumentalNeedsLead: return "Pick instruments below. The first one leads the track."
        }
    }

    // MARK: - Controls

    private func group<Content: View>(
        _ title: LocalizedStringKey, count: String? = nil, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title, bundle: .module).themedFont(.small, weight: .semibold)
                if let count { Text(verbatim: count).themedFont(.small).foregroundStyle(.appSecondary) }
            }
            .accessibilityElement(children: .combine).accessibilityAddTraits(.isHeader)
            content()
        }
    }

    private func genrePicker(_ title: LocalizedStringKey, _ binding: Binding<String?>) -> some View {
        Picker(selection: binding) {
            Text("None", bundle: .module).tag(String?.none)
            ForEach(MusicPromptCatalog.genres) { family in
                Section(family.name) {
                    ForEach(family.items, id: \.self) { Text(verbatim: $0).tag(String?.some($0)) }
                }
            }
        } label: { Text(title, bundle: .module) }
            .pickerStyle(.menu)
    }

    private func optionPicker(_ title: LocalizedStringKey, _ options: [String], _ binding: Binding<String?>) -> some View {
        Picker(selection: binding) {
            Text("None", bundle: .module).tag(String?.none)
            ForEach(options, id: \.self) { Text(verbatim: $0).tag(String?.some($0)) }
        } label: { Text(title, bundle: .module) }
            .pickerStyle(.menu)
    }

    private func multiChips(
        _ items: [String], selected: [String], limit: Int, set: @escaping ([String]) -> Void
    ) -> some View {
        FlowLayout(spacing: 6, lineSpacing: 6) {
            ForEach(items, id: \.self) { item in
                let on = selected.contains(item)
                chip(item, selected: on, disabled: !on && selected.count >= limit) {
                    set(MusicPromptSelection.toggled(selected, item, limit: limit))
                }
            }
        }
    }

    private func singleChips(_ items: [String], selected: String?, set: @escaping (String?) -> Void) -> some View {
        FlowLayout(spacing: 6, lineSpacing: 6) {
            ForEach(items, id: \.self) { item in
                chip(item, selected: item == selected) { set(item == selected ? nil : item) }
            }
        }
    }

    private func chip(_ label: String, selected: Bool, disabled: Bool = false, action: @escaping () -> Void) -> some View {
        AppChip(Text(verbatim: label), isOn: selected, action: action)
            .disabled(disabled)
    }
}
