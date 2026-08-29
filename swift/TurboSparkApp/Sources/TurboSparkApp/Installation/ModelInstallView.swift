import AppKit
import SwiftUI
import TurboSpark

/// Welcome and initial setup view displayed when no local models are detected.
/// Scans system hardware, displays memory limits, highlights MoE streaming advantages,
/// and presents hardware-tailored recommendations.
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

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "bolt.horizontal.circle.fill")
                .font(.system(size: 42))
                .foregroundStyle(TurboSparkTheme.accentColor)
                .accessibilityHidden(true)

            Text("Welcome to TurboSpark")
                .font(.system(.title2, design: .rounded).bold())
                .accessibilityAddTraits(.isHeader)

            Text("No local models found in ~/.turbospark. Choose a recommended model below to start.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }

    // MARK: - Recommendations Section

    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Recommended Models for Your Mac")
                        .font(.headline)
                    Text("Ranked by compatibility with your \(model.telemetry?.chip ?? "device") memory budget.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Picker("Filter", selection: $selectedFilter) {
                    ForEach(RecommendationFilter.allCases) { f in
                        Text(f.rawValue).tag(f)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 240)
                .accessibilityLabel("Recommendations filter")
            }

            if displayedRecommendations.isEmpty {
                VStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Calculating hardware fit recommendations...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
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
                Label("Rescan Storage", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .accessibilityHint("Rescans local model directories and reloads recommendations")

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Label("Choose Existing Model Folder...", systemImage: "folder")
            }
            .buttonStyle(.bordered)
            .accessibilityHint("Opens a folder picker for an already-installed model")

            Spacer()

            Button {
                model.openModelHub()
            } label: {
                Label("Browse Full Model Hub", systemImage: "square.grid.2x2")
            }
            .buttonStyle(.bordered)
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
        if let recs = try? TurboSparkCatalog.recommend() {
            self.recommendations = recs
        }
        isLoadingRecommendations = false
    }
}
