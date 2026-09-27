import Foundation

enum ImageResolutionPreset: String, CaseIterable, Identifiable, Hashable {
    case square512
    case square768
    case square1024
    case portrait4x3
    case landscape4x3
    case portrait16x9
    case landscape16x9

    var id: Self { self }

    var width: UInt32 {
        switch self {
        case .square512: 512
        case .square768: 768
        case .square1024: 1024
        case .portrait4x3: 768
        case .landscape4x3: 1024
        case .portrait16x9: 576
        case .landscape16x9: 1024
        }
    }

    var height: UInt32 {
        switch self {
        case .square512: 512
        case .square768: 768
        case .square1024: 1024
        case .portrait4x3: 1024
        case .landscape4x3: 768
        case .portrait16x9: 1024
        case .landscape16x9: 576
        }
    }

    var label: String { "\(width) x \(height)" }
}
