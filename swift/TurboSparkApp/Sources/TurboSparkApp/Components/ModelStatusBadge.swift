import AppKit
import SwiftUI
import TurboSpark

/// Interactive model chooser badge in the status HUD.
struct ModelStatusBadge: View {
    @ObservedObject var model: AppModel

    var body: some View {
        Menu {
            installedModelsSection
            Divider()
            Button {
                model.openModelHub()
            } label: {
                Label("Browse Model Hub…", systemImage: "square.grid.2x2")
            }
            .keyboardShortcut("m", modifiers: [.command])

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Label("Choose Model Folder…", systemImage: "folder")
            }
        } label: {
            HStack(spacing: 6) {
                statusDot
                Text(model.selected?.alias ?? (model.installed.isEmpty ? "No Model" : "Select Model"))
                    .font(.callout.weight(.semibold))
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
            }
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Switch active model or browse Model Hub (⌘M)")
        // The Menu's auto-derived label is the model's display text, but the
        // chevron and trailing label confuse VoiceOver. Force a clean label
        // so a screen reader says "Active model: Qwen 3.6, menu".
        .accessibilityLabel("Active model")
        .accessibilityValue(model.selected?.alias ?? (model.installed.isEmpty ? "No Model" : "Select Model"))
        .accessibilityHint("Switch active model or browse Model Hub")
    }

    @ViewBuilder
    private var installedModelsSection: some View {
        if model.installed.isEmpty {
            Text("No models installed yet").font(.caption)
        } else {
            Section("Installed Models") {
                ForEach(model.installed) { item in
                    Button {
                        model.openChatWithModel(item)
                    } label: {
                        HStack {
                            Text(item.alias)
                            Spacer()
                            Text("(\(item.family))")
                                .foregroundStyle(.secondary)
                            if model.selected?.alias == item.alias {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            }
        }
    }

    @Environment(\.accessibilityDifferentiateWithoutColor) private var differentiateWithoutColor

    @ViewBuilder
    private var statusDot: some View {
        if model.isInstallingModel || model.opening {
            ProgressView().controlSize(.mini)
        } else if model.session != nil {
            if differentiateWithoutColor {
                Image(systemName: "checkmark.circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.green)
                    .accessibilityHidden(true)
            } else {
                Circle().fill(.green).frame(width: 8, height: 8).accessibilityHidden(true)
            }
        } else if !model.installed.isEmpty {
            if differentiateWithoutColor {
                Image(systemName: "pause.circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .accessibilityHidden(true)
            } else {
                Circle().fill(.orange).frame(width: 8, height: 8).accessibilityHidden(true)
            }
        } else {
            if differentiateWithoutColor {
                Image(systemName: "minus.circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
            } else {
                Circle().fill(.gray).frame(width: 8, height: 8).accessibilityHidden(true)
            }
        }
    }
}

