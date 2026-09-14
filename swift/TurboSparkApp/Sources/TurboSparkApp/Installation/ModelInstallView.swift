import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Welcome and initial setup view displayed when no local models are detected.
/// Scans system hardware, displays memory limits, highlights MoE streaming advantages,
/// and presents hardware-tailored recommendations.
@MainActor
struct ModelInstallView: View {
    @ObservedObject var model: AppModel
    @State private var recommendations: [ModelRecommendation] = []
    @State private var isLoadingRecommendations = false
    @State private var selectedFilter: RecommendationFilter = .recommended

    enum RecommendationFilter: String, CaseIterable, Identifiable {
        case recommended = "Best Fit (MoE & Fast)"
        case all = "All Catalog Models"

        var id: String { rawValue }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                if ModelStorageManager.isLMStudioDirectoryPresent(customPath: model.lmStudioDirectory) {
                    lmStudioDetectedBanner
                }
                HardwareScanCard(telemetry: model.telemetry)
                MoESpotlightCard()

                if model.isInstallingModel {
                    ActiveInstallCard(model: model)
                }

                recommendationsSection
                utilityActionsRow
            }
            .frame(maxWidth: 680)
            .padding(.horizontal, 24)
            .padding(.vertical, 32)
            .frame(maxWidth: .infinity)
        }
        .task {
            loadRecommendations()
        }
    }

    // MARK: - Header & Identity

    private var lmStudioDetectedBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "shippingbox.fill")
                .themedFont(.title2)
                .foregroundStyle(Color.accentColor)

            VStack(alignment: .leading, spacing: 2) {
                Text("LM Studio Library Detected", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                Text("Found models at ~/.lmstudio/models. TurboSpark can run your LM Studio models directly without copying files.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            Spacer()

            Button {
                model.enableLMStudioDetection = true
                model.persistSettings()
                model.refreshModels()
                model.showToast("Scanned LM Studio models folder", style: .success)
            } label: { Text("Scan LM Studio", bundle: .module) }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
        .padding(12)
        .background(Color.accentColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.accentColor.opacity(0.3), lineWidth: 1))
    }

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "bolt.horizontal.circle.fill")
                .themedFont(.display)
                .foregroundStyle(.appAccent)
                .help("TurboSpark Local Inference")
                .accessibilityHidden(true)

            Text("Welcome to TurboSpark", bundle: .module)
                .themedFont(.title2, weight: .bold)
                .accessibilityAddTraits(.isHeader)

            Text("No local models found in ~/.turbospark. Choose a recommended model below to start.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
        }
    }

    // MARK: - Recommendations Section

    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Recommended Models for Your Mac", bundle: .module)
                        .themedFont(.base, weight: .semibold)
                    Text("Ranked by compatibility with your \(model.telemetry?.chip ?? "device") memory budget.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                Picker(selection: $selectedFilter) {
                    ForEach(RecommendationFilter.allCases) { f in
                        Text(f.rawValue).tag(f)
                    }
                } label: { Text("Filter", bundle: .module) }
                .pickerStyle(.segmented)
                .frame(width: 240)
                .accessibilityLabel("Recommendations filter")
            }

            if displayedRecommendations.isEmpty {
                VStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Calculating hardware fit recommendations...", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Calculating hardware fit recommendations")
            } else {
                VStack(spacing: 10) {
                    ForEach(displayedRecommendations) { rec in
                        ModelRecommendationRow(recommendation: rec, model: model)
                    }
                }
            }
        }
    }

    // MARK: - Bottom Utility Actions

    private var utilityActionsRow: some View {
        HStack(spacing: 12) {
            Button {
                model.refreshModels()
                loadRecommendations()
            } label: {
                Label { Text("Rescan Storage", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
            }
            .buttonStyle(.bordered)
            .help("Rescan storage for local models")
            .accessibilityHint("Rescans local model directories and reloads recommendations")

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Label { Text("Choose Existing Model Folder...", bundle: .module) } icon: { Image(systemName: "folder") }
            }
            .buttonStyle(.bordered)
            .help("Select an existing model directory")
            .accessibilityHint("Opens a folder picker for an already-installed model")

            Spacer()

            Button {
                model.openModelHub()
            } label: {
                Label { Text("Browse Full Model Hub", bundle: .module) } icon: { Image(systemName: "square.grid.2x2") }
            }
            .buttonStyle(.bordered)
            .help("Open the model catalog")
            .accessibilityHint("Opens the full model catalog in the sidebar")
        }
        .controlSize(.regular)
        .padding(.top, 4)
    }

    // MARK: - Helpers

    private var displayedRecommendations: [ModelRecommendation] {
        if selectedFilter == .recommended {
            // Prioritize runnable MoE and high-performance models
            return recommendations.filter { $0.runs }
        }
        return recommendations
    }

    private func loadRecommendations() {
        isLoadingRecommendations = true
        recommendations = model.fitRecommendations()
        isLoadingRecommendations = false
    }
}
