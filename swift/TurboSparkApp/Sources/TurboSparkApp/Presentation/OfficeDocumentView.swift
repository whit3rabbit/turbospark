import AppKit
import SwiftUI

/// Read-only viewer for docx, xlsx and pptx (ZP-08).
///
/// The viewer layer only ever READS: `OfficeRenderModel.load` opens the file
/// for reading and parses a private temp copy, and nothing in this folder
/// holds a writable handle to the source. Any failure (damaged, encrypted,
/// over a limit, mid-write) shows a bounded card above the system QuickLook
/// preview, so the worst a bad file costs is fidelity.
@MainActor
struct OfficeDocumentView: View {
    let url: URL
    /// Bumps when the file is rewritten in place (an artifact revision), so
    /// the same path re-parses instead of showing the previous bytes.
    var revision = 0

    private struct LoadKey: Hashable {
        let url: URL
        let revision: Int
    }

    private enum Phase {
        case loading
        case loaded(OfficeDocument)
        case failed(OfficeRenderError)
    }

    @State private var phase = Phase.loading

    var body: some View {
        VStack(spacing: 0) {
            switch phase {
            case .loading:
                VStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading...", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .failed(let error):
                OfficeRenderFailureView(url: url, error: error)
            case .loaded(let document):
                viewer(for: document)
                Divider()
                OfficeFooterView(notes: document.notes, hiddenSheets: hiddenSheets(document))
            }
        }
        .task(id: LoadKey(url: url, revision: revision)) { await load() }
    }

    @ViewBuilder
    private func viewer(for document: OfficeDocument) -> some View {
        switch document {
        case .docx(let value): DocxView(document: value)
        case .xlsx(let value): XlsxView(document: value)
        case .pptx(let value): PptxView(document: value)
        }
    }

    private func hiddenSheets(_ document: OfficeDocument) -> Int {
        if case .xlsx(let value) = document { return value.hiddenSheetCount }
        return 0
    }

    private func load() async {
        phase = .loading
        let target = url
        let result = await Task.detached(priority: .userInitiated) {
            Result { try OfficeRenderModel.load(url: target) }
        }.value
        guard !Task.isCancelled else { return }
        switch result {
        case .success(let document):
            phase = .loaded(document)
        case .failure(let error):
            phase = .failed((error as? OfficeRenderError) ?? .unreadable)
        }
    }
}

/// The bounded error card. It names the failing part (verbatim, capped), says
/// in one sentence what to do, and keeps the QuickLook preview below it.
struct OfficeRenderFailureView: View {
    let url: URL
    let error: OfficeRenderError

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundStyle(.appWarning)
                        .accessibilityHidden(true)
                    Text("This document could not be rendered here", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                }
                body(for: error.category)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let part = error.part {
                    Text(verbatim: part)
                        .themedCode(.tiny)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Button {
                    NSWorkspace.shared.open(url)
                } label: {
                    Text("Open with default app", bundle: .module)
                }
                .appAction(.secondary, size: .small)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .accessibilityElement(children: .contain)
            Divider()
            QuickLookPreviewView(url: url)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .accessibilityLabel(Text("Quick Look preview of \(url.lastPathComponent)", bundle: .module))
        }
    }

    private func body(for category: OfficeRenderError.Category) -> Text {
        switch category {
        case .overLimits:
            return Text("This file is over the size or time limits for the in-app viewer. The system preview is shown instead.", bundle: .module)
        case .damaged:
            return Text("This file is damaged, encrypted, or uses a structure the in-app viewer does not accept. The system preview is shown instead.", bundle: .module)
        }
    }
}

/// Read-only badge plus the documented limits and per-file notices.
struct OfficeFooterView: View {
    let notes: OfficeRenderNotes
    let hiddenSheets: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Label {
                    Text("Read-only", bundle: .module)
                        .themedFont(.tiny, weight: .medium)
                } icon: {
                    Image(systemName: "lock")
                        .themedFont(.tiny)
                }
                .foregroundStyle(.appSecondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .overlay(Capsule().stroke(.appBorder, lineWidth: 1))
                Spacer(minLength: 0)
            }
            Text("Simplified view: themes, charts, SmartArt and footnotes are not rendered, and layout can differ from Office.", bundle: .module)
            if notes.hasMacros {
                Text("Macros in this file are never run.", bundle: .module)
            }
            if notes.truncated {
                Text("Showing only the first part of this document.", bundle: .module)
            }
            if hiddenSheets > 0 {
                Text("Hidden sheets are not shown.", bundle: .module)
            }
        }
        .themedFont(.tiny)
        .foregroundStyle(.appSecondary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .fixedSize(horizontal: false, vertical: true)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }
}
