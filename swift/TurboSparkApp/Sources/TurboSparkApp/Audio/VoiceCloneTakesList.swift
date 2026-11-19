import SwiftUI

/// Every saved clone take, newest first: play it in place, see which voice
/// and quality made it, reuse its reference, save or delete it.
@MainActor
struct VoiceCloneTakesList: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        let takes = controller.speechTakes
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                AppSectionHeader(title: Text("Takes", bundle: .module))
                Text(verbatim: String(takes.count)).monospacedDigit().foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
            }
            if takes.isEmpty {
                Text("No outputs yet", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            } else {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(takes) { take in VoiceCloneTakeRow(controller: controller, take: take) }
                }
                .accessibilityRotor(Text("Takes", bundle: .module)) {
                    ForEach(takes) { take in
                        AccessibilityRotorEntry(take.title + ", " + audioTime(take.duration), id: take.id)
                    }
                }
            }
        }
    }
}

@MainActor
private struct VoiceCloneTakeRow: View {
    @ObservedObject var controller: AudioWorkspaceController
    let take: AudioLibraryItem

    @Environment(\.appTheme) private var theme
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @State private var hovered = false
    @State private var deletingItem: AudioLibraryItem?

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    private var isSelected: Bool { controller.selectedID == take.id }
    private var isPlaying: Bool { controller.playbackItemID == take.id && controller.isPlaying }
    private var reference: AudioVoiceReference? { take.recipe.speech?.reference }

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { playButton; info; Spacer(minLength: 8); actions }
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) { playButton; info }
                actions
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .background(isSelected ? theme.selection : Color.clear,
                    in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(edge, lineWidth: isSelected && theme.isHighContrast ? 2 : 1)
        }
        .onHover { hovered = $0 }
        .tsAnimation(TSMotion.hover, value: hovered)
        .tsAnimation(TSMotion.select, value: isSelected)
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
        .modifier(AudioDeleteConfirmation(controller: controller, item: $deletingItem))
    }

    private var edge: AnyShapeStyle {
        if isSelected { return AnyShapeStyle(.appAccent.opacity(theme.isHighContrast ? 1 : 0.45)) }
        return hovered ? AnyShapeStyle(.appBorder) : AnyShapeStyle(Color.clear)
    }

    private var playButton: some View {
        AudioTakeAuditionButton(controller: controller, item: take)
    }

    private var info: some View {
        Button { controller.select(take) } label: {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(verbatim: take.title).themedFont(.base, weight: isSelected ? .semibold : .regular)
                        .lineLimit(1).truncationMode(.tail)
                    if isPlaying {
                        // A live cue next to the title: the bars run only while
                        // this take plays, and hold still under Reduce Motion.
                        Image(systemName: "waveform")
                            .themedFont(.small).foregroundStyle(.appAccent)
                            .symbolEffect(.variableColor.iterative, isActive: !reduceMotion)
                            .accessibilityHidden(true)
                    }
                }
                HStack(spacing: 10) {
                    Text(take.createdAt, format: .dateTime.month(.abbreviated).day().hour().minute())
                    Text(verbatim: audioTime(take.duration)).monospacedDigit()
                    if let reference {
                        Label { Text(verbatim: reference.title).lineLimit(1).truncationMode(.middle) } icon: {
                            Image(systemName: "waveform")
                        }
                        .help(Text(verbatim: reference.title))
                    }
                    if let options = take.recipe.speech?.cloneOptions { qualityBadge(options.steps) }
                }
                .themedFont(.tiny).foregroundStyle(.appSecondary).lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityAction(named: Text("Reuse reference", bundle: .module)) { controller.useVoiceReference(fromTake: take) }
    }

    @ViewBuilder private func qualityBadge(_ steps: Int) -> some View {
        Group {
            if let preset = CloneQuality(steps: steps) { preset.titleText } else { Text("Steps: \(steps)", bundle: .module) }
        }
        .padding(.horizontal, 6).padding(.vertical, 1)
        .background(.appAccent.opacity(0.12), in: Capsule())
    }

    private var actions: some View {
        Menu {
            if reference != nil {
                Button { controller.useVoiceReference(fromTake: take) } label: { Text("Reuse reference", bundle: .module) }
                    .disabled(!controller.canEditVoiceReference)
            }
            Button { controller.exportAudio(take) } label: { Text("Save As...", bundle: .module) }
                .disabled(controller.hasRecordingActivity || controller.isExporting)
            Divider()
            Button(role: .destructive) { deletingItem = take } label: { Text("Delete", bundle: .module) }
                .disabled(!controller.canDelete(take))
        } label: {
            Image(systemName: "ellipsis")
        }
        .appMenuAction(size: .small, quiet: true)
        .menuIndicator(.hidden)
        .accessibilityLabel(Text("Audio actions", bundle: .module))
        .help(Text("Audio actions", bundle: .module))
    }
}
