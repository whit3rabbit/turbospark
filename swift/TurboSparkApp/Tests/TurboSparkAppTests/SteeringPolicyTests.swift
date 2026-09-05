import XCTest

@testable import TurboSparkApp
import TurboSpark

/// `AppSteeringPolicy` and the settings round trip for steering presets.
///
/// Every case here is pure: no session, no install, no Metal. That is the
/// point of the type existing (`swift/CLAUDE.md` Gotcha 26) -- the same
/// assertions written against `AppModel` would need a real model and would
/// therefore never run.
final class SteeringPolicyTests: XCTestCase {

    private func preset(
        hidden: Int? = 5120,
        spans: Int? = 64,
        path: String = "/vectors/d.gguf",
        mode: AppSteeringModeOption = .ablate,
        scale: Double = 0.3
    ) -> AppSteeringPreset {
        AppSteeringPreset(
            name: "ocean", vectorPath: path, mode: mode, scale: scale,
            vectorHidden: hidden, vectorSpannedLayers: spans
        )
    }

    // MARK: - Compatibility

    func testMatchingShapesAreAllowed() {
        let result = AppSteeringPolicy.compatibility(
            preset: preset(), modelHidden: 5120, modelLayers: 64)
        XCTAssertEqual(result, .shapeMatches)
        XCTAssertTrue(result.allowsEnabling)
    }

    /// The refusal has to carry BOTH numbers. "Incompatible" alone leaves a
    /// user with no idea whether the file or the model is the odd one out.
    func testAWidthMismatchIsRefusedAndNamesBothNumbers() {
        let result = AppSteeringPolicy.compatibility(
            preset: preset(hidden: 4096), modelHidden: 5120, modelLayers: 64)
        XCTAssertEqual(result, .widthMismatch(vector: 4096, model: 5120))
        XCTAssertFalse(result.allowsEnabling)
        XCTAssertTrue(result.summary.contains("4096"), result.summary)
        XCTAssertTrue(result.summary.contains("5120"), result.summary)
    }

    /// **THE SPAN IS COMPARED, NOT THE COUNT.** A vector carrying directions
    /// for blocks 1...63 covers 63 of them and SPANS 64. Comparing the count
    /// would call it compatible with a 63-layer model it overruns, which the
    /// engine then refuses at load.
    func testTheLayerCheckComparesTheSpanAndNotTheCoveredCount() {
        let overrun = AppSteeringPolicy.compatibility(
            preset: preset(spans: 64), modelHidden: 5120, modelLayers: 52)
        XCTAssertEqual(overrun, .layerOverrun(vectorSpans: 64, modelLayers: 52))
        XCTAssertFalse(overrun.allowsEnabling)

        let fits = AppSteeringPolicy.compatibility(
            preset: preset(spans: 52), modelHidden: 5120, modelLayers: 52)
        XCTAssertEqual(fits, .shapeMatches)
    }

    /// A width mismatch outranks a layer overrun, matching
    /// `SteeringSet::validate`'s own order: the width is the more fundamental
    /// fact and reporting the layer count beside it would bury it.
    func testAWidthMismatchIsReportedAheadOfALayerOverrun() {
        let result = AppSteeringPolicy.compatibility(
            preset: preset(hidden: 4096, spans: 999), modelHidden: 5120, modelLayers: 64)
        XCTAssertEqual(result, .widthMismatch(vector: 4096, model: 5120))
    }

    /// **AN UNREAD SHAPE IS `unknown`, NOT A MISMATCH AND NOT A ZERO**
    /// (Gotcha 23). A preset restored from a settings file written before the
    /// shape was recorded has no width, and rendering that as 0 would show a
    /// mismatch against every model.
    func testAnUnreadShapeIsUnknownRatherThanZeroWidth() {
        let result = AppSteeringPolicy.compatibility(
            preset: preset(hidden: nil), modelHidden: 5120, modelLayers: 64)
        guard case .unknown = result else { return XCTFail("expected unknown, got \(result)") }
        XCTAssertTrue(
            result.allowsEnabling,
            "unknown must stay usable, or every preset from an older settings file is dead"
        )
        XCTAssertFalse(result.summary.contains("0 "), result.summary)
    }

    func testAModelWithNoDeclaredHiddenSizeIsUnknownRatherThanAMismatch() {
        let result = AppSteeringPolicy.compatibility(
            preset: preset(), modelHidden: nil, modelLayers: nil)
        guard case .unknown = result else { return XCTFail("expected unknown, got \(result)") }
    }

    func testAPresetWithNoVectorPathSteersNothing() {
        let blank = AppSteeringPolicy.compatibility(
            preset: preset(path: "   "), modelHidden: 5120, modelLayers: 64)
        XCTAssertEqual(blank, .noVector)
        XCTAssertFalse(blank.allowsEnabling)
        XCTAssertEqual(
            AppSteeringPolicy.compatibility(preset: nil, modelHidden: 5120, modelLayers: 64),
            .noVector
        )
    }

    /// **`.shapeMatches` MUST NOT CLAIM THE VECTOR IS RIGHT FOR THIS MODEL.**
    /// `docs/OBLITERATION.md` states the hazard plainly: a vector extracted
    /// for a different checkpoint of the same width opens, steers, and moves
    /// behaviour in a direction nobody asked for, silently. Shape is the only
    /// thing checkable, so shape is the only thing claimed.
    func testAShapeMatchDoesNotClaimSemanticCompatibility() {
        let summary = AppSteeringPolicy.Compatibility.shapeMatches.summary
        XCTAssertTrue(summary.lowercased().contains("shape"), summary)
        XCTAssertFalse(
            summary.lowercased().contains("compatible"),
            "a shape match is not compatibility, and the word invites the wrong conclusion"
        )
        XCTAssertTrue(
            summary.lowercased().contains("another checkpoint"),
            "the caveat has to be in the string a user actually reads: \(summary)"
        )
    }

    // MARK: - Disabled reason

    /// An unsupported family reports the ENGINE's own wording, so the reason
    /// a control is grey is the reason the load would give.
    func testAnUnsupportedFamilyReportsTheEnginesOwnReason() {
        let reason = AppSteeringPolicy.disabledReason(
            familySupported: false,
            familyReason: "steering is not wired for family Qwen4Exp",
            preset: preset(),
            compatibility: .shapeMatches
        )
        XCTAssertEqual(reason, "steering is not wired for family Qwen4Exp")
    }

    /// The family check outranks everything: a mismatched vector on a family
    /// that cannot steer at all should say the family, not the width.
    func testTheFamilyCheckOutranksTheShapeCheck() {
        let reason = AppSteeringPolicy.disabledReason(
            familySupported: false,
            familyReason: "not wired for family Qwen4Exp",
            preset: preset(hidden: 4096),
            compatibility: .widthMismatch(vector: 4096, model: 5120)
        )
        XCTAssertEqual(reason?.contains("Qwen4Exp"), true)
    }

    /// An unsupported family with no reason still says something. A grey
    /// control with an empty tooltip is the "missing feature" reading Gotcha
    /// 33 exists to avoid.
    func testAnUnsupportedFamilyWithNoReasonStillExplainsItself() {
        let reason = AppSteeringPolicy.disabledReason(
            familySupported: false, familyReason: nil,
            preset: preset(), compatibility: .shapeMatches
        )
        XCTAssertNotNil(reason)
        XCTAssertFalse(reason!.isEmpty)
    }

    /// **NO SESSION MEANS USABLE**, not disabled. A control that is dead
    /// until a model happens to be loaded is dead in the state a user reaches
    /// for it in.
    func testWithNoSessionTheControlIsUsable() {
        XCTAssertNil(
            AppSteeringPolicy.disabledReason(
                familySupported: nil, familyReason: nil,
                preset: preset(), compatibility: .shapeMatches
            )
        )
    }

    func testNoPresetAndNoVectorEachExplainThemselves() {
        XCTAssertNotNil(
            AppSteeringPolicy.disabledReason(
                familySupported: true, familyReason: nil,
                preset: nil, compatibility: .noVector
            )
        )
        XCTAssertNotNil(
            AppSteeringPolicy.disabledReason(
                familySupported: true, familyReason: nil,
                preset: preset(path: ""), compatibility: .noVector
            )
        )
    }

    // MARK: - Needs reload

    /// **STEERING RESOLVES ONCE, AT OPEN.** Turning it on changes nothing
    /// about the model already loaded, and this is what lets the UI say so
    /// instead of showing a switch that did nothing.
    private func sessionSteering(
        active: Bool, mode: String? = nil, scale: Double? = nil
    ) throws -> SessionInfo.Steering {
        // Built in pieces rather than as one interpolated literal: nesting
        // interpolation inside a multiline string blows the macOS 14 SDK
        // type-checker's budget (`swift/CLAUDE.md` Gotcha 45).
        let activeText = active ? "true" : "false"
        var modeText = "null"
        if let mode { modeText = "\"" + mode + "\"" }
        var scaleText = "null"
        if let scale { scaleText = String(scale) }
        let json =
            "{ \"active\": " + activeText
            + ", \"supported\": true, \"reason\": null"
            + ", \"mode\": " + modeText
            + ", \"scale\": " + scaleText
            + ", \"summary\": null }"
        return try JSONDecoder().decode(SessionInfo.Steering.self, from: Data(json.utf8))
    }

    func testTurningSteeringOnWithoutReloadingNeedsAReload() throws {
        XCTAssertTrue(
            AppSteeringPolicy.needsReload(
                wantEnabled: true, wantPreset: preset(),
                sessionSteering: try sessionSteering(active: false)
            )
        )
    }

    /// **BOTH DIRECTIONS**, which is the half that is easy to miss: turning
    /// steering OFF also does nothing until reload, and a UI that only
    /// flagged the on-direction would show a model as unsteered while it is
    /// still steering.
    func testTurningSteeringOffWithoutReloadingAlsoNeedsAReload() throws {
        XCTAssertTrue(
            AppSteeringPolicy.needsReload(
                wantEnabled: false, wantPreset: preset(),
                sessionSteering: try sessionSteering(active: true, mode: "ablate", scale: 0.3)
            )
        )
    }

    func testAMatchingSessionNeedsNoReload() throws {
        XCTAssertFalse(
            AppSteeringPolicy.needsReload(
                wantEnabled: true, wantPreset: preset(mode: .ablate, scale: 0.3),
                sessionSteering: try sessionSteering(active: true, mode: "ablate", scale: 0.3)
            )
        )
    }

    func testAChangedModeOrScaleNeedsAReload() throws {
        XCTAssertTrue(
            AppSteeringPolicy.needsReload(
                wantEnabled: true, wantPreset: preset(mode: .renorm, scale: 0.3),
                sessionSteering: try sessionSteering(active: true, mode: "ablate", scale: 0.3)
            )
        )
        XCTAssertTrue(
            AppSteeringPolicy.needsReload(
                wantEnabled: true, wantPreset: preset(scale: 0.8),
                sessionSteering: try sessionSteering(active: true, mode: "ablate", scale: 0.3)
            )
        )
    }

    /// The scale makes a round trip through JSON and an f32 in the engine, so
    /// an exact `Double` comparison would report a pending reload forever.
    func testAScaleThatOnlyDiffersByFloatRoundingNeedsNoReload() throws {
        XCTAssertFalse(
            AppSteeringPolicy.needsReload(
                wantEnabled: true, wantPreset: preset(scale: 0.3),
                sessionSteering: try sessionSteering(
                    active: true, mode: "ablate", scale: 0.30000000000000004)
            )
        )
    }

    /// With no session there is nothing to be out of step WITH.
    func testWithNoSessionNothingNeedsReloading() {
        XCTAssertFalse(
            AppSteeringPolicy.needsReload(
                wantEnabled: true, wantPreset: preset(), sessionSteering: nil)
        )
    }

    /// Enabled with a preset that has no vector is not "on": the open would
    /// steer nothing, so the session correctly reports inactive and there is
    /// no reload owed.
    func testEnabledWithNoVectorIsNotAPendingChange() throws {
        XCTAssertFalse(
            AppSteeringPolicy.needsReload(
                wantEnabled: true, wantPreset: preset(path: ""),
                sessionSteering: try sessionSteering(active: false)
            )
        )
    }

    // MARK: - The default scale

    /// **0.3 AND NOT 1.0, and it is a measurement.**
    /// `docs/OBLITERATION.md` records ablation at alpha 1.0 over every layer
    /// COLLAPSING the turn on a real 27B install: immediate end-of-turn, no
    /// text. A default that lands on a documented failure mode is worse than
    /// no default, which is the same correction the steering probe in that
    /// repo had to make to its own default.
    func testTheDefaultScaleIsTheMeasuredSafeOneRatherThanFullStrength() {
        XCTAssertEqual(AppSteeringPreset.defaultScale, 0.3)
        XCTAssertEqual(AppSteeringPreset().scale, 0.3)
    }

    // MARK: - Persistence

    /// **ONE BAD PRESET COSTS THAT PRESET, NEVER THE SETTINGS FILE**
    /// (state#45, Gotcha 13). `decodeLenient([T].self, ...)` would catch the
    /// throw at the ARRAY level and fall back to `[]`, discarding every good
    /// preset -- read by a user as the app losing their configuration.
    func testOneUndecodablePresetDoesNotTakeTheOthersWithIt() throws {
        let json = """
            {
              "steeringPresets": [
                { "id": "\(UUID().uuidString)", "name": "good", "vectorPath": "/a.gguf",
                  "mode": "ablate", "scale": 0.3, "layers": "", "target": 0, "gate": 0,
                  "notes": "", "vectorHidden": 5120, "vectorSpannedLayers": 64 },
                "this is not a preset at all",
                { "id": "\(UUID().uuidString)", "name": "also good", "vectorPath": "/b.gguf",
                  "mode": "renorm", "scale": 0.5, "layers": "20:45", "target": 0, "gate": 0,
                  "notes": "", "vectorHidden": 4096, "vectorSpannedLayers": 32 }
              ],
              "steeringEnabled": true
            }
            """
        let settings = try JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))
        XCTAssertEqual(settings.steeringPresets.count, 2)
        XCTAssertEqual(settings.steeringPresets.map(\.name), ["good", "also good"])
        XCTAssertEqual(settings.steeringPresets[1].mode, .renorm)
        XCTAssertTrue(settings.steeringEnabled)
    }

    /// **A NON-SCALAR JUNK ELEMENT MUST NOT HANG THE LOAD.**
    /// `decodeLenientElements` skips a failing element by consuming it as a
    /// throwaway, and if that consume ever fails to ADVANCE the unkeyed
    /// container the loop spins on it forever -- at app launch, inside
    /// `loadSettings`, with no error and no window. A lost preset is a bad
    /// outcome; a hang is a much worse one, so every shape a JSON array can
    /// hold is exercised here rather than just the string the case above
    /// uses. The timeout is the assertion.
    func testJunkElementsOfEveryShapeAreSkippedWithoutHanging() throws {
        let good =
            "{ \"id\": \"\(UUID().uuidString)\", \"name\": \"good\","
            + " \"vectorPath\": \"/a.gguf\", \"mode\": \"ablate\" }"
        let json =
            "{ \"steeringPresets\": ["
            + "\"a string\", 42, true, null, [1, 2, 3], [], {}, "
            + good
            + "] }"

        let expectation = XCTestExpectation(description: "decode finishes")
        var decoded: MacAppSettings?
        DispatchQueue.global().async {
            decoded = try? JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))
            expectation.fulfill()
        }
        XCTAssertEqual(XCTWaiter().wait(for: [expectation], timeout: 5), .completed)

        let presets = try XCTUnwrap(decoded).steeringPresets
        // `{}` decodes to a defaulted preset, because every field on
        // `AppSteeringPreset` is optional by design (that is what stops one
        // bad FIELD costing the row). So the empty object survives as a blank
        // and the genuinely unparseable shapes are dropped.
        XCTAssertTrue(
            presets.contains { $0.name == "good" },
            "the valid preset must survive whatever precedes it: \(presets.map(\.name))"
        )
        XCTAssertLessThanOrEqual(presets.count, 2, "only {} and the good row may decode")
    }

    /// An unknown MODE raw value falls back rather than throwing, so a
    /// settings file written by a build that grew a fifth mode still loads.
    func testAnUnknownModeFallsBackInsteadOfDroppingThePreset() throws {
        let json = """
            {
              "steeringPresets": [
                { "id": "\(UUID().uuidString)", "name": "future", "vectorPath": "/a.gguf",
                  "mode": "somethingNew", "scale": 0.4 }
              ]
            }
            """
        let settings = try JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))
        XCTAssertEqual(settings.steeringPresets.count, 1)
        XCTAssertEqual(settings.steeringPresets[0].mode, .ablate)
        XCTAssertEqual(settings.steeringPresets[0].scale, 0.4)
    }

    func testPresetsRoundTripThroughTheSettingsFile() throws {
        let original = MacAppSettings(
            steeringPresets: [preset()],
            activeSteeringPresetID: "abc",
            steeringEnabled: true
        )
        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(restored.steeringPresets, original.steeringPresets)
        XCTAssertEqual(restored.activeSteeringPresetID, "abc")
        XCTAssertTrue(restored.steeringEnabled)
    }

    /// **STEERING IS OFF BY DEFAULT.** A settings file that predates presets
    /// must not come back steering, and the default has to be reachable
    /// through `MacAppSettings` rather than only through a property
    /// declaration `AppModel.init` overwrites (Gotcha 39).
    func testSteeringIsOffByDefaultAndCarriesNoPresets() throws {
        let settings = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertFalse(settings.steeringEnabled)
        XCTAssertTrue(settings.steeringPresets.isEmpty)
        XCTAssertEqual(settings.activeSteeringPresetID, "")
    }

    // MARK: - Server guardrails

    /// The app's guardrails setting has to reach the SERVED path, which is a
    /// different enforcement point from the agent loop's own engine.
    func testServerGuardrailsFollowTheAppSetting() {
        XCTAssertEqual(AppModel.serverGuardrails(from: .alwaysOff), .off)
        XCTAssertEqual(AppModel.serverGuardrails(from: .alwaysOn), .on)
        // `.select` means "decide per project or per chat" and a server
        // request has neither, so the engine's own default is the honest
        // fallback rather than off.
        XCTAssertEqual(AppModel.serverGuardrails(from: .select), .on)
    }

    // MARK: - Tooltips

    /// **AN INERT CONTROL EXPLAINS ITSELF FIRST.** A user looking at a greyed
    /// pill wants to know why it is grey before anything else, so the inert
    /// reason outranks both the global-mode text and the on/off text.
    func testTheInertReasonOutranksEveryOtherGuardrailsTooltip() {
        let text = ForgeGuardrailsPillControl.helpText(
            inertReason: "This chat sends no tools",
            isGlobalFixed: true,
            modeLabel: "Always On",
            isEnabled: true,
            nativeReason: "the Mistral dialect defines no tool-call markup"
        )
        XCTAssertEqual(text, "This chat sends no tools")
    }

    /// **A CHECKPOINT WITH NO TOOL MARKUP IS A REASON TO LEAVE GUARDRAILS ON,
    /// NOT A REASON TO HIDE THE CONTROL.** The tooltip has to say that, or a
    /// user reads "no native tool calls" as "tools do not work here" and
    /// turns off the one thing making them work.
    func testANonNativeCheckpointReadsAsAReasonToKeepGuardrails() {
        let text = ForgeGuardrailsPillControl.helpText(
            inertReason: nil,
            isGlobalFixed: false,
            modeLabel: "Auto",
            isEnabled: true,
            nativeReason: "the Mistral dialect defines no tool-call markup"
        )
        XCTAssertTrue(text.contains("Mistral"), text)
        XCTAssertTrue(
            text.contains("rescue"),
            "the tooltip must say what makes tool calls work here: \(text)"
        )
    }

    func testANativeCheckpointAddsNoCaveatToTheGuardrailsTooltip() {
        let text = ForgeGuardrailsPillControl.helpText(
            inertReason: nil, isGlobalFixed: false, modeLabel: "Auto",
            isEnabled: false, nativeReason: nil
        )
        XCTAssertFalse(text.contains("rescue"), text)
        XCTAssertTrue(text.contains("click to enable"), text)
    }

    /// **PENDING OUTRANKS "ON".** Without this the pill says On while the
    /// loaded model is unsteered, which is the exact silent no-op this
    /// feature exists to make visible.
    func testAPendingReloadOutranksTheOnState() {
        let text = SteeringPillControl.helpText(
            disabledReason: nil, isOn: true, needsReload: true,
            summary: "ablate at alpha 0.3 over 63 of 64 layers"
        )
        XCTAssertTrue(text.lowercased().contains("reload"), text)
        XCTAssertFalse(text.contains("Running:"), text)
    }

    /// When it IS running, the engine's own summary is what is shown --
    /// read back rather than restated from the settings.
    func testARunningEditReportsTheEnginesOwnSummary() {
        let text = SteeringPillControl.helpText(
            disabledReason: nil, isOn: true, needsReload: false,
            summary: "ablate at alpha 0.3 over 63 of 64 layers"
        )
        XCTAssertTrue(text.contains("ablate at alpha 0.3 over 63 of 64 layers"), text)
    }

    func testADisabledSteeringPillOutranksEverything() {
        let text = SteeringPillControl.helpText(
            disabledReason: "not wired for family Qwen4Exp",
            isOn: true, needsReload: true, summary: "ablate"
        )
        XCTAssertEqual(text, "not wired for family Qwen4Exp")
    }

    // MARK: - Server status

    /// A server started before this app tracked the value reports UNKNOWN
    /// rather than "on". Saying "on" would be a guess rendered as a reading
    /// (`swift/CLAUDE.md` Gotcha 23), and the value cannot change without a
    /// restart, so the honest row tells the user what to do about it.
    func testAnUntrackedServerReportsUnknownGuardrailsRatherThanOn() throws {
        let rows = ServerStatusRows(info: try serverInfo(), guardrails: nil)
        XCTAssertTrue(rows.guardrailsLabel.contains("unknown"), rows.guardrailsLabel)
        XCTAssertTrue(rows.guardrailsLabel.contains("restart"), rows.guardrailsLabel)
    }

    func testTheGuardrailsRowFollowsWhatTheServerStartedWith() throws {
        XCTAssertEqual(
            ServerStatusRows(info: try serverInfo(), guardrails: .off).guardrailsLabel, "off")
        XCTAssertEqual(
            ServerStatusRows(info: try serverInfo(), guardrails: .on).guardrailsLabel, "on")
    }

    private func serverInfo() throws -> ServerInfo {
        let json =
            "{ \"port\": 8080, \"host\": \"127.0.0.1\", \"modelId\": \"m\","
            + " \"models\": [\"m\"], \"authEnabled\": false, \"uptimeSeconds\": 1 }"
        return try JSONDecoder().decode(ServerInfo.self, from: Data(json.utf8))
    }

}
