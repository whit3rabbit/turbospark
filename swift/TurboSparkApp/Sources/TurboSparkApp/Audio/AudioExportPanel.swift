import AppKit
import UniformTypeIdentifiers

enum AudioExportPreset: String, CaseIterable, Sendable {
    case nativeWAV, videoWAV, flac, aiff, m4a

    var fileExtension: String {
        switch self {
        case .nativeWAV, .videoWAV: return "wav"
        case .flac: return "flac"
        case .aiff: return "aiff"
        case .m4a: return "m4a"
        }
    }

    var title: String {
        switch self {
        case .nativeWAV: return AppStrings.localized("WAV (original format)")
        case .videoWAV: return AppStrings.localized("WAV (48 kHz for video)")
        case .flac: return "FLAC"
        case .aiff: return "AIFF"
        case .m4a: return AppStrings.localized("M4A")
        }
    }
}

/// Keep format choice beside the filename and folder in the system save dialog.
@MainActor
final class AudioExportPanel: NSObject {
    let panel = NSSavePanel()
    let formatPicker = NSPopUpButton(frame: .zero, pullsDown: false)
    private(set) var preset: AudioExportPreset

    init(title: String, preset: AudioExportPreset = .nativeWAV) {
        self.preset = preset
        super.init()
        panel.title = AppStrings.localized("Save as Audio...")
        panel.prompt = AppStrings.localized("Save")
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false
        panel.allowedContentTypes = [UTType(filenameExtension: preset.fileExtension) ?? .audio]
        panel.nameFieldStringValue = ProfileBackup.sanitizedFileName(title, fallback: "Audio") + "." + preset.fileExtension
        formatPicker.addItems(withTitles: AudioExportPreset.allCases.map(\.title))
        formatPicker.selectItem(at: AudioExportPreset.allCases.firstIndex(of: preset)!)
        formatPicker.target = self
        formatPicker.action = #selector(changeFormat)
        formatPicker.setAccessibilityLabel(AppStrings.localized("Format"))
        let label = NSTextField(labelWithString: AppStrings.localized("Format"))
        let accessory = NSStackView(views: [label, formatPicker])
        accessory.spacing = 12
        accessory.edgeInsets = NSEdgeInsets(top: 12, left: 0, bottom: 12, right: 0)
        accessory.setFrameSize(accessory.fittingSize)
        panel.accessoryView = accessory
    }

    @objc private func changeFormat() {
        select(AudioExportPreset.allCases[formatPicker.indexOfSelectedItem])
    }

    func select(_ value: AudioExportPreset) {
        let name = panel.nameFieldStringValue as NSString
        let stem = name.pathExtension.lowercased() == preset.fileExtension ? name.deletingPathExtension : name as String
        preset = value
        formatPicker.selectItem(at: AudioExportPreset.allCases.firstIndex(of: value)!)
        panel.allowedContentTypes = [UTType(filenameExtension: value.fileExtension) ?? .audio]
        panel.nameFieldStringValue = stem + "." + value.fileExtension
    }
}
