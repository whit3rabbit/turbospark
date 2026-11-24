import XCTest
@testable import TurboSparkApp

/// The saved microphone is kept by Core Audio UID, because the numeric
/// AudioObjectID changes across reboots and replugs. A UID that no longer
/// resolves falls back to the system default and says so, never to some
/// other device.
final class MicrophoneSelectionTests: XCTestCase {
    private let builtIn = MicrophoneSelection.Device(id: 41, uid: "BuiltInMicrophoneDevice", name: "MacBook Pro Microphone")
    private let usb = MicrophoneSelection.Device(id: 97, uid: "AppleUSBAudioEngine:Shure:MV7:1", name: "Shure MV7")

    func testNoSavedChoiceUsesTheSystemDefault() {
        let resolved = MicrophoneSelection.resolve(savedUID: nil, available: [builtIn, usb])
        XCTAssertNil(resolved.deviceID)
        XCTAssertFalse(resolved.savedDeviceMissing)
    }

    func testASavedUIDResolvesToTheCurrentObjectID() {
        let resolved = MicrophoneSelection.resolve(savedUID: usb.uid, available: [builtIn, usb])
        XCTAssertEqual(resolved.deviceID, 97)
        XCTAssertFalse(resolved.savedDeviceMissing)
        // Same device, new object id after a replug.
        let replugged = MicrophoneSelection.Device(id: 130, uid: usb.uid, name: usb.name)
        XCTAssertEqual(MicrophoneSelection.resolve(savedUID: usb.uid, available: [builtIn, replugged]).deviceID, 130)
    }

    func testAGoneDeviceFallsBackToTheDefaultWithANote() {
        let resolved = MicrophoneSelection.resolve(savedUID: usb.uid, available: [builtIn])
        XCTAssertNil(resolved.deviceID, "it must never pick a different device")
        XCTAssertTrue(resolved.savedDeviceMissing)
        let empty = MicrophoneSelection.resolve(savedUID: usb.uid, available: [])
        XCTAssertNil(empty.deviceID)
        XCTAssertTrue(empty.savedDeviceMissing)
    }

    func testABlankSavedUIDIsTheDefault() {
        let resolved = MicrophoneSelection.resolve(savedUID: "", available: [builtIn])
        XCTAssertNil(resolved.deviceID)
        XCTAssertFalse(resolved.savedDeviceMissing)
    }

    // MARK: Level

    func testRMSOfSilenceAndOfAFullScaleSquare() {
        XCTAssertEqual(MicrophoneLevel.rms([]), 0)
        XCTAssertEqual(MicrophoneLevel.rms([0, 0, 0]), 0)
        XCTAssertEqual(MicrophoneLevel.rms([1, -1, 1, -1]), 1, accuracy: 1e-6)
        XCTAssertEqual(MicrophoneLevel.rms([0.5, -0.5]), 0.5, accuracy: 1e-6)
        XCTAssertEqual(MicrophoneLevel.rms([.nan, 1]), 0, "a bad buffer reads as silence")
    }

    func testDecibelsMapLinearlyOntoTheBarFromTheFloorToFullScale() {
        XCTAssertEqual(MicrophoneLevel.fraction(rms: 1), 1, accuracy: 1e-6)
        XCTAssertEqual(MicrophoneLevel.fraction(rms: 0), 0)
        // -30 dBFS sits halfway along a -60 dB floor.
        XCTAssertEqual(MicrophoneLevel.fraction(rms: 0.0316228), 0.5, accuracy: 1e-3)
        XCTAssertEqual(MicrophoneLevel.fraction(rms: 0.001), 0, accuracy: 1e-6, "-60 dBFS is the floor")
        XCTAssertEqual(MicrophoneLevel.fraction(rms: 0.0001), 0, "below the floor clamps")
        XCTAssertEqual(MicrophoneLevel.fraction(rms: 4), 1, "above full scale clamps")
        XCTAssertEqual(MicrophoneLevel.fraction(rms: .nan), 0)
        XCTAssertEqual(MicrophoneLevel.fraction(rms: -0.5), 0)
    }

    func testTheMeterSaysItHearsYouOnlyAboveRoomNoise() {
        XCTAssertFalse(MicrophoneLevel.hearsInput(fraction: 0))
        XCTAssertFalse(MicrophoneLevel.hearsInput(fraction: MicrophoneLevel.fraction(rms: 0.003)), "about -50 dBFS is room noise")
        XCTAssertTrue(MicrophoneLevel.hearsInput(fraction: MicrophoneLevel.fraction(rms: 0.03)), "about -30 dBFS is speech")
    }

    // MARK: Settings

    func testOldSettingsWithoutTheKeyStillLoadAsTheDefault() throws {
        XCTAssertNil(MacAppSettings().voiceCallMicrophoneUID)
        let old = try JSONDecoder().decode(MacAppSettings.self, from: Data(#"{"voiceCallEnabled": true}"#.utf8))
        XCTAssertNil(old.voiceCallMicrophoneUID)
        XCTAssertTrue(old.voiceCallEnabled, "the rest of an old file still loads")
        XCTAssertNil(
            try JSONDecoder().decode(MacAppSettings.self, from: Data(#"{"voiceCallMicrophoneUID": 7}"#.utf8))
                .voiceCallMicrophoneUID, "a wrong type must not fail the file")
        var settings = MacAppSettings()
        settings.voiceCallMicrophoneUID = usb.uid
        let back = try JSONDecoder().decode(MacAppSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(back.voiceCallMicrophoneUID, usb.uid)
    }
}
