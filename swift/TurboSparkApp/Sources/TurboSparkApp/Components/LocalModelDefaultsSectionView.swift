import SwiftUI
import TurboSpark

/// Local model defaults: the floor under an automatically-sized context
/// window, and how much of the machine a model may commit when it loads.
///
/// Its own file rather than another section inside `ModelsSettingsPaneView`,
/// which is already near the 400-line guideline (`swift/CLAUDE.md` Gotcha 15).
///
/// The policy these two controls drive lives in the engine, not here: see
/// `docs/LOAD_GUARD.md` for the tiers, what each one reserves, and why
/// `relaxed` is the default.
public struct LocalModelDefaultsSectionView: View {
    @ObservedObject var model: AppModel

    /// The slider's stops. Powers of two rather than a continuous range,
    /// because a context window is only ever read back at one of these and a
    /// floor of 13,000 tokens would be a number nobody chose.
    private static let contextStops: [UInt32] = [
        0, 2048, 4096, 8192, 16384, 32768, 65536, 131_072,
    ]

    /// The custom tier's stops, in bytes.
    private static let ceilingStops: [UInt64] = [
        1 << 30, 2 << 30, 3 << 30, 4 << 30, 6 << 30, 8 << 30, 12 << 30, 16 << 30, 24 << 30,
        32 << 30,
    ]

    public init(model: AppModel) {
        self.model = model
    }

    private var floorIndex: Binding<Double> {
        Binding(
            get: {
                let current = model.runtimeOptions.minAutoContextTokens
                let idx = Self.contextStops.firstIndex(of: current)
                    ?? Self.contextStops.lastIndex(where: { $0 <= current })
                    ?? 0
                return Double(idx)
            },
            set: { newValue in
                let idx = min(max(Int(newValue.rounded()), 0), Self.contextStops.count - 1)
                model.runtimeOptions.minAutoContextTokens = Self.contextStops[idx]
                model.persistSettings()
            }
        )
    }

    private var ceilingIndex: Binding<Double> {
        Binding(
            get: {
                let current = model.runtimeOptions.loadGuardCustomBytes
                let idx = Self.ceilingStops.lastIndex(where: { $0 <= current }) ?? 3
                return Double(idx)
            },
            set: { newValue in
                let idx = min(max(Int(newValue.rounded()), 0), Self.ceilingStops.count - 1)
                model.runtimeOptions.loadGuardCustomBytes = Self.ceilingStops[idx]
                model.persistSettings()
            }
        )
    }

    private var floorLabel: String {
        let tokens = model.runtimeOptions.minAutoContextTokens
        return tokens == 0 ? "No minimum" : tokens.formatted()
    }

    private var ceilingLabel: String {
        let bytes = model.runtimeOptions.loadGuardCustomBytes
        guard bytes > 0 else { return "Not set" }
        return "\(bytes / (1 << 30)) GB"
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            modelDefaultsSection
            loadGuardSection
        }
    }

    private var modelDefaultsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Model defaults", bundle: .module)
                .themedFont(.base, weight: .semibold)

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Minimum AutoFit context length", bundle: .module)
                .settingsControl("Minimum AutoFit context length", pane: .models, timing: .modelReload)
                        .themedFont(.base)
                    Spacer()
                    Text(floorLabel)
                        .themedFont(.base).monospacedDigit()
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                // Says what it constrains AND what it does not. The setting is
                // named for AutoFit because it is scoped to it: an explicitly
                // chosen context length is never refused for being small.
                Text(
                    "Require AutoFit to provide at least this many tokens of context. "
                        + "Models that cannot meet this minimum will fail to load. "
                        + "An explicitly chosen context length is not affected."
                )
                .themedFont(.small)
                .foregroundStyle(.appSecondary)

                Slider(
                    value: floorIndex,
                    in: 0...Double(Self.contextStops.count - 1),
                    step: 1
                )
                .accessibilityLabel("Minimum AutoFit context length")
                .accessibilityValue(floorLabel)
            }
            .padding(14)
            .background(Color.secondary.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private var loadGuardSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Model loading guardrails", bundle: .module)
                .settingsControl("Model loading guardrails", pane: .models, timing: .modelReload)
                .themedFont(.base, weight: .semibold)

            VStack(alignment: .leading, spacing: 12) {
                Text(
                    "Loading models beyond system resource limits may cause instability. "
                        + "Guardrails hold memory back for the rest of the machine. "
                        + "Relaxed is the default and is what every published memory "
                        + "figure for this engine was measured under."
                )
                .themedFont(.small)
                .foregroundStyle(.appSecondary)

                ForEach(AppLoadGuardOption.allCases) { option in
                    guardRow(option)
                }

                if model.runtimeOptions.loadGuard == .custom {
                    customCeiling
                }
            }
            .padding(14)
            .background(Color.secondary.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private func guardRow(_ option: AppLoadGuardOption) -> some View {
        Button {
            model.runtimeOptions.loadGuard = option
            model.persistSettings()
        } label: {
            HStack(alignment: .top, spacing: 10) {
                Image(
                    systemName: model.runtimeOptions.loadGuard == option
                        ? "largecircle.fill.circle" : "circle"
                )
                .foregroundStyle(model.runtimeOptions.loadGuard == option ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(option.menuLabel)
                        .themedFont(.base)
                        .foregroundStyle(.appText)
                    Text(option.detailText)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(model.runtimeOptions.loadGuard == option ? [.isSelected] : [])
    }

    private var customCeiling: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Maximum allocation", bundle: .module)
                .settingsControl("Maximum allocation", pane: .models, timing: .modelReload)
                    .themedFont(.base)
                Spacer()
                Text(ceilingLabel)
                    .themedFont(.base).monospacedDigit()
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.secondary.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            // The distinction that makes this number usable: it caps what the
            // engine ALLOCATES, not the size of the file on disk. A large
            // model streaming its experts is what this engine is for, and a
            // cap read against the install would refuse models that run fine.
            Text(
                "Caps what the engine allocates (expert cache plus KV), not the size of "
                    + "the model on disk. Models larger than this can still run by "
                    + "streaming from storage."
            )
            .themedFont(.small)
            .foregroundStyle(.appSecondary)

            Slider(
                value: ceilingIndex,
                in: 0...Double(Self.ceilingStops.count - 1),
                step: 1
            )
            .accessibilityLabel("Maximum allocation")
            .accessibilityValue(ceilingLabel)
        }
        .padding(.top, 4)
    }
}
