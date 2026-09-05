import AppKit
import SwiftUI
import TurboSpark

/// Right-column detail inspector for an installed model in the Model Manager.
struct InstalledModelDetailPaneView: View {
    @ObservedObject var model: AppModel
    let installedModel: InstalledModel
    @ObservedObject private var orgStore = ModelOrganizationStore.shared

    @State private var showingDeleteConfirm = false
    @State private var isEditingNotes = false
    @State private var notesText: String = ""
    @State private var isAddingTag = false
    @State private var newTagText: String = ""

    private var descriptor: ModelFeatureDescriptor {
        ModelFeatureDescriptor.resolve(installedModel: installedModel)
    }

    private var visuals: ModelFamilyVisuals {
        ModelFamilyVisuals.resolve(
            alias: installedModel.alias,
            family: installedModel.family,
            name: installedModel.alias
        )
    }

    private var isCurrentlyLoaded: Bool {
        model.selected?.path == installedModel.path && model.session != nil
    }

    private var isCurrentlySelected: Bool {
        model.selected?.alias == installedModel.alias
    }

    private var isFavorite: Bool {
        orgStore.isFavorite(alias: installedModel.alias, path: installedModel.path)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                heroHeader
                actionBar
                featureBadgesStrip
                memoryFitCard
                technicalSpecsCard
                organizationCard
                developerCommandsCard
            }
            .padding(24)
        }
        .onAppear {
            notesText = orgStore.notes(alias: installedModel.alias, path: installedModel.path)
        }
        .onChange(of: installedModel.alias) {
            notesText = orgStore.notes(alias: installedModel.alias, path: installedModel.path)
            isAddingTag = false
        }
        .confirmationDialog(
            "Delete \(installedModel.alias)?",
            isPresented: $showingDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button("Delete Model", role: .destructive) {
                model.deleteModel(installedModel)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will permanently remove the model files (\(MetricFormat.storage(installedModel.installBytes))) from disk at:\n\(installedModel.path)")
        }
    }

    // MARK: - Hero Header

    private var heroHeader: some View {
        HStack(alignment: .top, spacing: 16) {
            ModelLogoView(visuals: visuals, size: 56, cornerRadius: 14)

            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 8) {
                    Text(installedModel.alias)
                        .font(.title2.weight(.bold))
                        .lineLimit(2)

                    Text("INSTALLED")
                        .font(.system(size: 9, weight: .bold))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.teal.opacity(0.16), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(Color.teal)

                    Button {
                        orgStore.toggleFavorite(alias: installedModel.alias, path: installedModel.path)
                    } label: {
                        Image(systemName: isFavorite ? "star.fill" : "star")
                            .font(.system(size: 14))
                            .foregroundStyle(isFavorite ? Color.yellow : Color.secondary)
                    }
                    .buttonStyle(.plain)
                    .help(isFavorite ? "Remove from Favorites" : "Mark as Favorite")
                    .accessibilityLabel(isFavorite ? "Favorited" : "Favorite")

                    Spacer(minLength: 4)

                    statusBadge
                }

                HStack(spacing: 8) {
                    HStack(spacing: 4) {
                        Image(systemName: visuals.iconSystemName)
                            .font(.caption2)
                            .foregroundStyle(visuals.accentColor)
                        Text(visuals.family)
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                    Text("\u{2022}")
                        .foregroundStyle(.tertiary)

                    Text(MetricFormat.storage(installedModel.installBytes))
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)

                    Text("\u{2022}")
                        .foregroundStyle(.tertiary)

                    Text(descriptor.storageSource.shortLabel)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                }

                if !installedModel.repo.isEmpty && installedModel.repo != "local" {
                    if let url = URL(string: "https://huggingface.co/\(installedModel.repo)") {
                        Link(destination: url) {
                            HStack(spacing: 3) {
                                Text(installedModel.repo)
                                Image(systemName: "arrow.up.right.square")
                                    .font(.system(size: 9))
                            }
                            .font(.caption.monospaced())
                            .foregroundStyle(Color.accentColor)
                        }
                        .help("Open \(installedModel.repo) on Hugging Face: \(url.absoluteString)")
                        .accessibilityLabel("Hugging Face repository \(installedModel.repo)")
                        .accessibilityHint("Opens this model's page on Hugging Face in your web browser")
                        .accessibilityAddTraits(.isLink)
                    } else {
                        Text(installedModel.repo)
                            .font(.caption.monospaced())
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var statusBadge: some View {
        if isCurrentlyLoaded {
            HStack(spacing: 4) {
                Circle().fill(Color.green).frame(width: 7, height: 7)
                Text("Loaded in Memory")
                    .font(.caption2.weight(.bold))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(Color.green.opacity(0.16), in: Capsule())
            .foregroundStyle(Color.green)
        } else if isCurrentlySelected {
            HStack(spacing: 4) {
                Circle().fill(Color.accentColor).frame(width: 6, height: 6)
                Text("Selected")
                    .font(.caption2.weight(.semibold))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(Color.accentColor.opacity(0.14), in: Capsule())
            .foregroundStyle(Color.accentColor)
        } else {
            HStack(spacing: 4) {
                Circle().fill(Color.secondary).frame(width: 5, height: 5)
                Text("Ready to Load")
                    .font(.caption2.weight(.medium))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3.5)
            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.4), in: Capsule())
            .foregroundStyle(.secondary)
        }
    }

    // MARK: - Action Bar

    private var actionBar: some View {
        HStack(spacing: 10) {
            if isCurrentlyLoaded {
                Button {
                    model.unloadModel()
                } label: {
                    Label("Unload", systemImage: "eject.fill")
                        .font(.system(size: 12, weight: .medium))
                        .frame(height: 28)
                        .padding(.horizontal, 12)
                }
                .buttonStyle(.plain)
                .background(Color.orange.opacity(0.16), in: RoundedRectangle(cornerRadius: 6))
                .foregroundStyle(Color.orange)
                .disabled(!model.canUnloadModel)
                .help("Eject model and release unified memory")
            } else {
                Button {
                    model.selectModel(installedModel)
                } label: {
                    Label("Load Model", systemImage: "bolt.fill")
                        .font(.system(size: 12, weight: .medium))
                        .frame(height: 28)
                        .padding(.horizontal, 12)
                }
                .buttonStyle(.plain)
                .background(Color.accentColor.opacity(0.18), in: RoundedRectangle(cornerRadius: 6))
                .foregroundStyle(Color.accentColor)
                .help("Load model into memory slot cache for inference")
            }

            Button {
                model.openChatWithModel(installedModel)
            } label: {
                Label("Start Chat", systemImage: "bubble.left.and.bubble.right.fill")
                    .font(.system(size: 12, weight: .medium))
                    .frame(height: 28)
                    .padding(.horizontal, 12)
            }
            .buttonStyle(.plain)
            .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 6))
            .foregroundStyle(.white)
            .help("Switch to Chat view with this model")

            Button {
                ModelStorageManager.revealInFinder(path: installedModel.path)
            } label: {
                Label("Reveal", systemImage: "folder")
                    .font(.system(size: 12, weight: .medium))
                    .frame(height: 28)
                    .padding(.horizontal, 10)
            }
            .buttonStyle(.plain)
            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            .foregroundStyle(.primary)
            .help("Show model files in Finder: \(installedModel.path)")
            .accessibilityLabel("Reveal in Finder")
            .accessibilityHint("Opens Finder showing the model files at \(installedModel.path)")
            .accessibilityAddTraits(.isLink)

            Spacer()

            Button(role: .destructive) {
                showingDeleteConfirm = true
            } label: {
                Label("Delete", systemImage: "trash")
                    .font(.system(size: 12, weight: .medium))
                    .frame(height: 28)
                    .padding(.horizontal, 10)
            }
            .buttonStyle(.plain)
            .background(Color.red.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
            .foregroundStyle(Color.red)
            // state#73: this pane offered Delete during a load, where the
            // model file is mapped by an open still in flight.
            .disabled(!model.canDeleteModel)
            .help("Delete model files from disk")
        }
    }

    // MARK: - Feature Badges Strip

    private var featureBadgesStrip: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model Highlights & Features")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            FlowLayout(spacing: 6, lineSpacing: 6) {
                // MoE vs Dense
                if descriptor.routingType == .moe {
                    ModelFeatureBadgeView.moe(details: descriptor.routingDetails, style: .regular)
                } else {
                    ModelFeatureBadgeView.dense(details: "Dense Transformer", style: .regular)
                }

                // Quantization format
                ModelFeatureBadgeView.quant(descriptor.quantFormat, style: .regular)

                // Drafter
                if descriptor.speculativeDrafter == .dynamicSloth {
                    ModelFeatureBadgeView.dynamicSloth(style: .regular)
                } else if descriptor.speculativeDrafter == .mtp {
                    ModelFeatureBadgeView.mtp(style: .regular)
                }

                // Weight container
                ModelFeatureBadgeView.format(descriptor.format, style: .regular)

                // Live Steering
                if descriptor.isSteeringReady {
                    ModelFeatureBadgeView.steeringReady(style: .regular)
                }

                // Linear Attention
                if descriptor.hasLinearAttention {
                    ModelFeatureBadgeView.linearAttention(style: .regular)
                }

                // Chunked Prefill
                if descriptor.supportsChunkedPrefill {
                    ModelFeatureBadgeView(
                        title: "Chunked Prefill",
                        iconSystemName: "rectangle.split.3x1.fill",
                        tintColor: .teal,
                        tooltip: "Chunked GPU prefill for long prompts without watchdog timeouts",
                        style: .regular
                    )
                }

                // Reasoning
                if descriptor.supportsReasoning {
                    ModelFeatureBadgeView(
                        title: "Reasoning Channel",
                        iconSystemName: "brain.head.profile",
                        tintColor: .indigo,
                        tooltip: "Supports chain-of-thought and internal reasoning extraction",
                        style: .regular
                    )
                }

                // Tool Calls
                if descriptor.supportsToolCalls {
                    ModelFeatureBadgeView(
                        title: "Tool Guardrails",
                        iconSystemName: "hammer.fill",
                        tintColor: .orange,
                        tooltip: "Supports Forge Tool-Call Guardrails with argument validation and nudging",
                        style: .regular
                    )
                }

                // Vision
                if descriptor.supportsVision {
                    ModelFeatureBadgeView(
                        title: "Vision Pipeline",
                        iconSystemName: "eye.fill",
                        tintColor: .pink,
                        tooltip: "Multi-modal vision pipeline with mRoPE positional dispatch",
                        style: .regular
                    )
                }

                // Storage source
                ModelFeatureBadgeView.source(descriptor.storageSource, style: .regular)
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Memory Fit Card

    private var memoryFitCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Unified Memory & Working Set", systemImage: "memorychip")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                // `estimatedWorkingSetRAM` is nil when there is no real
                // measurement or on-disk size to derive it from (U3): show
                // that honestly rather than a placeholder byte count.
                if let ram = descriptor.estimatedWorkingSetRAM {
                    Text("RAM requirement: ~\(MetricFormat.storage(ram))")
                        .font(.caption.monospacedDigit().weight(.medium))
                        .foregroundStyle(Color.accentColor)
                } else {
                    Text("RAM requirement: unknown")
                        .font(.caption.monospacedDigit().weight(.medium))
                        .foregroundStyle(.secondary)
                }
            }

            if descriptor.isSlotCacheStreaming {
                Text("This is a Mixture-of-Experts (MoE) model. TurboSpark streams routed expert weights on-demand from SSD storage through a lean working-memory slot cache (~1.6 to 2.2 GB peak RAM), allowing it to run smoothly on any Mac configuration.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("This is a Dense model. Dense tokens touch every weight layer, so weights are mapped directly into unified memory (\(MetricFormat.storage(installedModel.installBytes))).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Technical Specifications

    private var technicalSpecsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Technical Specifications", systemImage: "info.circle")
                .font(.subheadline.weight(.semibold))

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    Text("Architecture").font(.caption).foregroundStyle(.secondary)
                    Text(visuals.architectureType).font(.caption.weight(.medium))
                }
                GridRow {
                    Text("Quantization").font(.caption).foregroundStyle(.secondary)
                    Text(descriptor.quantFormat).font(.caption.weight(.medium))
                }
                GridRow {
                    Text("Format & Package").font(.caption).foregroundStyle(.secondary)
                    Text(descriptor.format.rawValue).font(.caption.weight(.medium))
                }
                if let ctx = descriptor.contextLimit {
                    GridRow {
                        Text("Trained Context").font(.caption).foregroundStyle(.secondary)
                        Text("\(ctx) tokens").font(.caption.monospacedDigit().weight(.medium))
                    }
                }
                if let layers = descriptor.layerCount {
                    GridRow {
                        Text("Layers").font(.caption).foregroundStyle(.secondary)
                        Text("\(layers) layers").font(.caption.monospacedDigit().weight(.medium))
                    }
                }
                if let hidden = descriptor.hiddenSize {
                    GridRow {
                        Text("Hidden Dimension").font(.caption).foregroundStyle(.secondary)
                        Text("\(hidden)").font(.caption.monospacedDigit().weight(.medium))
                    }
                }
                if let vocab = descriptor.vocabSize {
                    GridRow {
                        Text("Vocabulary Size").font(.caption).foregroundStyle(.secondary)
                        Text("\(vocab)").font(.caption.monospacedDigit().weight(.medium))
                    }
                }
                GridRow {
                    Text("Disk Path").font(.caption).foregroundStyle(.secondary)
                    Button {
                        ModelStorageManager.revealInFinder(path: installedModel.path)
                    } label: {
                        Text(installedModel.path)
                            .font(.caption.monospaced())
                            .lineLimit(2)
                            .foregroundStyle(Color.accentColor)
                    }
                    .buttonStyle(.plain)
                    .help("Reveal in Finder: \(installedModel.path)")
                    .accessibilityLabel("Disk path: \(installedModel.path)")
                    .accessibilityHint("Opens Finder to reveal model files")
                    .accessibilityAddTraits(.isLink)
                }
                if !installedModel.installedOn.isEmpty {
                    GridRow {
                        Text("Installed On").font(.caption).foregroundStyle(.secondary)
                        Text(installedModel.installedOn).font(.caption.weight(.medium))
                    }
                }
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - User Organization Card

    private var organizationCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Organization & Notes", systemImage: "tag")
                .font(.subheadline.weight(.semibold))

            // Tags section
            VStack(alignment: .leading, spacing: 6) {
                Text("Custom Tags")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                FlowLayout(spacing: 6, lineSpacing: 6) {
                    ForEach(orgStore.tags(alias: installedModel.alias, path: installedModel.path), id: \.self) { tag in
                        HStack(spacing: 4) {
                            Text(tag)
                                .font(.caption2.weight(.medium))
                            Button {
                                orgStore.removeTag(tag, for: installedModel.alias, path: installedModel.path)
                            } label: {
                                Image(systemName: "xmark")
                                    .font(.system(size: 8, weight: .bold))
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.accentColor.opacity(0.14), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(Color.accentColor)
                    }

                    if isAddingTag {
                        HStack(spacing: 4) {
                            TextField("Tag name", text: $newTagText)
                                .textFieldStyle(.plain)
                                .font(.caption2)
                                .frame(width: 80)
                                .onSubmit {
                                    orgStore.addTag(newTagText, for: installedModel.alias, path: installedModel.path)
                                    newTagText = ""
                                    isAddingTag = false
                                }
                            Button("Add") {
                                orgStore.addTag(newTagText, for: installedModel.alias, path: installedModel.path)
                                newTagText = ""
                                isAddingTag = false
                            }
                            .font(.caption2)
                            .buttonStyle(.plain)
                            .disabled(newTagText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 4))
                        .overlay { RoundedRectangle(cornerRadius: 4).stroke(Color.accentColor, lineWidth: 0.5) }
                    } else {
                        Button {
                            isAddingTag = true
                        } label: {
                            Label("Add Tag", systemImage: "plus")
                                .font(.caption2.weight(.medium))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 3)
                        }
                        .buttonStyle(.plain)
                        .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(.secondary)
                    }
                }
            }

            Divider()

            // Notes section
            VStack(alignment: .leading, spacing: 6) {
                Text("Personal Notes")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                TextField("Add personal notes for this model (e.g., best for Swift, fast test runner)...", text: $notesText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.caption)
                    .padding(8)
                    .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                    .overlay { RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor), lineWidth: 0.5) }
                    .onChange(of: notesText) { _, newValue in
                        orgStore.setNotes(newValue, for: installedModel.alias, path: installedModel.path)
                    }
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - CLI Developer Commands

    private var developerCommandsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Terminal / CLI Commands", systemImage: "terminal")
                .font(.subheadline.weight(.semibold))

            cliSnippet(
                title: "Run CLI REPL Chat",
                command: "turbospark-check --model \(installedModel.alias) --chat"
            )

            cliSnippet(
                title: "Serve OpenAI & Anthropic API",
                command: serveCommand
            )
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    /// The server launch line, carrying the app's own default system prompt so
    /// a server the user starts from here behaves like the app does.
    ///
    /// `--system` is a STARTUP flag on that binary: there is no control route,
    /// so a prompt changed here reaches a server only on its next launch.
    private var serveCommand: String {
        let base = "turbospark-server --model \(installedModel.alias)"
        let prompt = model.defaultSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return base }
        return "\(base) --system \(ShellQuote.single(prompt))"
    }

    private func cliSnippet(title: String, command: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            HStack {
                Text(command)
                    .font(.caption.monospaced())
                    .lineLimit(1)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(command, forType: .string)
                    model.showToast("Command copied to clipboard", style: .info)
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(.caption)
                }
                .buttonStyle(.plain)
                .help("Copy command")
            }
            .padding(6)
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 4))
        }
    }
}
