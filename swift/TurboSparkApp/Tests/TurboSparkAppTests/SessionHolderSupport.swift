import Foundation
import XCTest

@testable import TurboSparkApp

// Holders for the eject teardown suites. The real `TurboSparkSession` needs a
// multi-gigabyte checkpoint, so a probe object stands in for it, captured by
// the REAL `AppModel` task slots (`titleGenerationTask`, `memoryCaptureTask`,
// `backgroundAgentTasks`, `runTask`) that hold the session in the app.

/// Stands in for the engine session: whatever captures it keeps it alive, the
/// way a background agent keeps the real session (and its mapped weights).
final class SessionProbe {}

/// An owner holding a probe, like `AppModel.session` holds the session.
final class ProbeHolder {
    var thing: SessionProbe?
}

/// Starts holders of one probe and makes sure none outlives the test.
@MainActor
final class SessionHolders {
    /// Every gate a test opened or left closed, so a failed test cannot leave a
    /// holder spinning for the rest of the run.
    private var gates: [EjectBox<Bool>] = []

    func openAll() {
        for gate in gates { gate.value = true }
        gates = []
    }

    /// Opens `gate` at the end of the test even if an assertion failed first.
    func gateForCleanup(_ gate: EjectBox<Bool>) {
        gates.append(gate)
    }

    /// A sleep that does NOT return early when the task is cancelled, so a
    /// holder that ignores cancellation cannot spin hot.
    static func nap() async {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(3)) { done.resume() }
        }
    }

    /// Loops until the gate opens, or, for a holder that honors cancellation,
    /// until the task is cancelled. Keeps `probe` alive until it ends.
    static func spin(_ probe: SessionProbe, gate: EjectBox<Bool>, honorsCancel: Bool) async {
        while !gate.value && !(honorsCancel && Task.isCancelled) { await nap() }
        withExtendedLifetime(probe) {}
    }

    /// Starts a title task, a memory task and a running background agent that
    /// all hold one probe. Returns only a weak reference, so the test itself
    /// never keeps it alive; with `owner`, the owner holds it too (the way
    /// `AppModel.session` does). With a `gate` the holders ignore cancellation
    /// and end only when the gate opens (a request that will not let go).
    /// `chatID` is the chat the background agent belongs to: an agent with no
    /// chat never injects a notification, so the injection tests pass the
    /// selected chat.
    func hold(
        on model: AppModel, chatID: UUID? = nil, owner: ProbeHolder? = nil,
        ignoringCancelUntil ignoreGate: EjectBox<Bool>? = nil
    ) -> WeakObject {
        let probe = SessionProbe()
        owner?.thing = probe
        let gate = ignoreGate ?? EjectBox(false)
        gates.append(gate)
        let honorsCancel = ignoreGate == nil
        model.titleGenerationTask = Task { @MainActor [probe, weak model] in
            await Self.spin(probe, gate: gate, honorsCancel: honorsCancel)
            model?.titleGenerationTask = nil
        }
        model.memoryCaptureTask = Task { @MainActor [probe, weak model] in
            await Self.spin(probe, gate: gate, honorsCancel: honorsCancel)
            model?.memoryCaptureTask = nil
        }
        let id = "bga_t1"
        model.backgroundAgentRuns[id] = SubagentRunState(
            id: id, mode: .background, chatID: chatID, agentName: "t", displayName: "T",
            taskDescription: "t", promptHead: "t")
        let run = Task<SubagentRunResult, Never> { [probe] in
            await Self.spin(probe, gate: gate, honorsCancel: honorsCancel)
            return SubagentRunResult(
                agentName: "t", status: Task.isCancelled ? "cancelled" : "completed", finalResponse: "",
                totalTurns: 0, totalToolCalls: 0, durationSeconds: 0)
        }
        model.backgroundAgentTasks[id] = run
        // The real completion watcher of `launchBackgroundAgent`.
        Task { @MainActor [weak model] in
            let result = await run.value
            await model?.completeBackgroundAgent(id: id, result: result)
        }
        return WeakObject(probe)
    }

    /// A title task that, once cancelled, starts a successor (as the title
    /// queue does from the tail of a task), which holds the probe too. Counts
    /// the successors it started.
    func holdByTitleTaskWithSuccessor(on model: AppModel, successors: EjectBox<Int>) -> WeakObject {
        let probe = SessionProbe()
        // Opened by `openAll`, so a test that fails before the holders end
        // cannot leave them spinning on the main actor for the rest of the run.
        let gate = EjectBox(false)
        gates.append(gate)
        model.titleGenerationTask = Task { @MainActor [probe, weak model] in
            await Self.spin(probe, gate: gate, honorsCancel: true)
            successors.value += 1
            model?.titleGenerationTask = Task { @MainActor [probe, weak model] in
                await Self.spin(probe, gate: gate, honorsCancel: true)
                model?.titleGenerationTask = nil
            }
        }
        return WeakObject(probe)
    }

    /// A running turn (`generating`, `runTask`) that holds the probe and ends
    /// when cancelled, as `finishTurnTail` ends one. With `restarts`, the tail
    /// starts another turn in the same hop, as a queued prompt does.
    func holdByTurn(on model: AppModel, owner: ProbeHolder? = nil, restarts: Bool = false) -> WeakObject {
        let probe = SessionProbe()
        owner?.thing = probe
        let gate = EjectBox(false)
        gates.append(gate)
        func start(restarts: Bool) {
            model.generating = true
            model.isCancellationPending = false
            model.runTask = Task { @MainActor [probe] in
                await Self.spin(probe, gate: gate, honorsCancel: true)
                model.generating = false
                model.runTask = nil
                model.isCancellationPending = false
                if restarts { start(restarts: false) }
            }
        }
        start(restarts: restarts)
        return WeakObject(probe)
    }

    /// One task that holds a probe held by `holder` as well.
    func holdProbe(in holder: ProbeHolder, gate: EjectBox<Bool>) -> Task<Void, Never> {
        let probe = SessionProbe()
        holder.thing = probe
        gates.append(gate)
        return Task { @MainActor [probe] in await Self.spin(probe, gate: gate, honorsCancel: false) }
    }

    /// A chat session slot over a probe holder: loaded while the holder holds
    /// the probe, and `clear` is the owner letting go.
    static func slot(over holder: ProbeHolder, path: String) -> ChatSessionSlot {
        ChatSessionSlot(
            loadedPath: { holder.thing == nil ? nil : path },
            object: { holder.thing },
            clear: { holder.thing = nil })
    }
}
