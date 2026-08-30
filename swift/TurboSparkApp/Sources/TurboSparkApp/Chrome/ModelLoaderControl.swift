import AppKit
import SwiftUI
import TurboSpark

/// The top bar's model loader: pick a model, load it, eject it.
///
/// Load and eject are separate hit targets rather than one toggle, because the
/// two have different costs and an accidental eject discards a warm session
/// (weights, KV cache and expert slots) that takes seconds to rebuild.
struct ModelLoaderControl: View {
    @ObservedObject var model: AppModel

    @ScaledMetric private var barHeight: CGFloat = 28

    var body: some View {
        HStack(spacing: 0) {
            chooser
            if model.isReasoningSupported {
                separator
                reasoningPicker
            }
            if model.session != nil {
                separator
                ejectButton
            } else if model.canLoadModel {
                separator
                loadButton
            }
        }
        .frame(height: barHeight)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay {
            Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
        }
        .fixedSize()
    }

    private var reasoningPicker: some View {
        Menu {
            Section("Reasoning Effort") {
                ForEach(GenerateOptions.Reasoning.allCases) { level in
                    Button {
                        model.setReasoning(level)
                    } label: {
                        if model.reasoning == level {
                            Label(level.label + " - " + level.descriptionText, systemImage: "checkmark")
                        } else {
                            Text(level.label + " - " + level.descriptionText)
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: model.reasoning != .off ? "brain.head.profile" : "brain")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(model.reasoning != .off ? TurboSparkTheme.accentColor : Color.secondary)

                Text(model.reasoning != .off ? model.reasoning.label : "Off")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(model.reasoning != .off ? Color.primary : Color.secondary)

                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 7, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 8)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Reasoning effort: \(model.reasoning.label). Click to change (takes effect immediately without reloading model).")
        .accessibilityLabel("Reasoning effort: \(model.reasoning.label)")
        .accessibilityHint("Select reasoning depth")
    }

    private var separator: some View {
        Rectangle()
            .fill(TurboSparkTheme.hairlineColor)
            .frame(width: 0.5, height: barHeight * 0.6)
    }

    private var chooser: some View {
        Menu {
            installedModelsSection

            Divider()

            Button {
                model.activeSection = .modelHub
            } label: {
                Label("Browse models…", systemImage: "shippingbox")
            }

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Label("Open model folder…", systemImage: "folder")
            }

            if model.canReloadModel {
                Divider()
                Button {
                    model.reloadModel()
                } label: {
                    Label("Reload model", systemImage: "arrow.clockwise")
                }
            }
        } label: {
            HStack(spacing: 7) {
                statusIndicator
                VStack(alignment: .leading, spacing: 0) {
                    Text(primaryText)
                        .font(.system(size: 12, weight: .semibold))
                        .lineLimit(1)
                        .foregroundStyle(.primary)
                    if let secondaryText {
                        Text(secondaryText)
                            .font(.system(size: 9))
                            .lineLimit(1)
                            .foregroundStyle(.secondary)
                    }
                }
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .padding(.leading, 10)
            .padding(.trailing, 9)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .frame(maxWidth: 300)
        .help("Choose the active model")
        .accessibilityLabel("Active model")
        .accessibilityValue(primaryText)
        .accessibilityHint("Choose which installed model to use")
    }

    private var loadButton: some View {
        Button {
            model.loadModel()
        } label: {
            Label("Load", systemImage: "play.fill")
                .font(.system(size: 11, weight: .semibold))
                .labelStyle(.titleAndIcon)
                .imageScale(.small)
                .foregroundStyle(TurboSparkTheme.accentColor)
                .padding(.horizontal, 10)
                .frame(maxHeight: .infinity)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Load \(model.selected?.alias ?? "the selected model") into memory")
        .accessibilityLabel("Load model")
    }

    private var ejectButton: some View {
        Button {
            model.unloadModel()
        } label: {
            Image(systemName: "eject.fill")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 10)
                .frame(maxHeight: .infinity)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!model.canUnloadModel)
        .help("Eject the loaded model and free its memory")
        .accessibilityLabel("Eject model")
        .accessibilityHint("Unloads the model and frees its memory")
    }

    @ViewBuilder
    private var installedModelsSection: some View {
        if model.installed.isEmpty {
            Text("No models installed yet")
        } else {
            Section("Installed") {
                ForEach(model.installed) { item in
                    Button {
                        model.openChatWithModel(item)
                    } label: {
                        if model.selected?.alias == item.alias {
                            Label("\(item.alias)  (\(item.family))", systemImage: "checkmark")
                        } else {
                            Text("\(item.alias)  (\(item.family))")
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var statusIndicator: some View {
        if model.opening || model.isInstallingModel {
            ProgressView().controlSize(.mini).scaleEffect(0.7).frame(width: 8)
                .help("Loading or installing model")
        } else {
            Circle()
                .fill(indicatorColor)
                .frame(width: 7, height: 7)
                .help(statusTooltipText)
                .accessibilityHidden(true)
        }
    }

    private var statusTooltipText: String {
        if model.session != nil { return "Model loaded in memory" }
        if !model.installed.isEmpty { return "Model ready to load" }
        return "No models installed"
    }

    private var indicatorColor: Color {
        if model.session != nil { return .green }
        if !model.installed.isEmpty { return .orange }
        return .gray
    }

    private var primaryText: String {
        if model.installed.isEmpty { return "No model installed" }
        guard let selected = model.selected else { return "Select a model to load" }
        return selected.alias
    }

    private var secondaryText: String? {
        if model.opening { return "Loading…" }
        guard let info = model.info else {
            return model.selected == nil ? nil : "Not loaded"
        }
        return "\(info.maxContext.formatted(.number.notation(.compactName))) ctx"
    }
}
