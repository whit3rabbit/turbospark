import Foundation

struct WorkflowInterpreterLimits: Sendable {
    var maximumExecutedNodes: Int
    var maximumFacadeOperations: Int
    var maximumActors: Int
    var maximumLoopItems: Int

    init(
        maximumExecutedNodes: Int = 10_000,
        maximumFacadeOperations: Int = 500,
        maximumActors: Int = 100,
        maximumLoopItems: Int = 100
    ) {
        self.maximumExecutedNodes = maximumExecutedNodes
        self.maximumFacadeOperations = maximumFacadeOperations
        self.maximumActors = maximumActors
        self.maximumLoopItems = maximumLoopItems
    }
}

/// Engine-facing operations contain canonical values only. Command dispatch
/// carries a pinned key and named values, never an executable or argv template.
enum WorkflowInterpreterOperation: Equatable, Sendable {
    case ask(actor: String, prompt: WorkflowCanonicalValue, shape: WorkflowCanonicalValue)
    case join(graph: WorkflowCanonicalValue)
    case criticLoop(policy: WorkflowCanonicalValue)
    case worldRead(operation: WorkflowCanonicalValue)
    case run(commandKey: String, values: [String: WorkflowCanonicalValue])
    case report(value: WorkflowCanonicalValue)
    case artifact(value: WorkflowCanonicalValue)
    case phase(name: String)
}

enum WorkflowCancellationReason: Equatable, Sendable {
    case requested
    case deadlineExceeded
}

/// Shared with the active engine operation so cancellation and deadlines cross
/// the interpreter boundary without exposing host capabilities.
final class WorkflowCancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var reason: WorkflowCancellationReason?
    private var waiters: [CheckedContinuation<WorkflowCancellationReason, Never>] = []
    private var cancellationObservers: [UUID: @Sendable (WorkflowCancellationReason) -> Void] = [:]

    var cancellationReason: WorkflowCancellationReason? {
        lock.lock()
        defer { lock.unlock() }
        return reason
    }

    var isCancelled: Bool {
        cancellationReason != nil
    }

    func waitUntilCancelled() async -> WorkflowCancellationReason {
        await withCheckedContinuation { continuation in
            lock.lock()
            let currentReason = reason
            if currentReason == nil {
                waiters.append(continuation)
            }
            lock.unlock()

            if let currentReason {
                continuation.resume(returning: currentReason)
            }
        }
    }

    fileprivate func cancel(_ reason: WorkflowCancellationReason) {
        lock.lock()
        guard self.reason == nil else {
            lock.unlock()
            return
        }
        self.reason = reason
        let continuations = waiters
        waiters.removeAll()
        let observers = Array(cancellationObservers.values)
        cancellationObservers.removeAll()
        lock.unlock()

        for continuation in continuations {
            continuation.resume(returning: reason)
        }
        for observer in observers {
            observer(reason)
        }
    }

    func observeCancellation(
        _ observer: @escaping @Sendable (WorkflowCancellationReason) -> Void
    ) -> UUID? {
        lock.lock()
        if let reason {
            lock.unlock()
            observer(reason)
            return nil
        }
        let id = UUID()
        cancellationObservers[id] = observer
        lock.unlock()
        return id
    }

    func removeCancellationObserver(_ id: UUID) {
        lock.lock()
        cancellationObservers.removeValue(forKey: id)
        lock.unlock()
    }
}

struct WorkflowAttemptContext: Sendable {
    let cancellation: WorkflowCancellationToken
    let deadline: ContinuousClock.Instant?
}

protocol WorkflowCommandSink: Sendable {
    func perform(
        _ operation: WorkflowInterpreterOperation,
        at site: WorkflowSiteKey?,
        context: WorkflowAttemptContext
    ) async throws -> WorkflowCanonicalValue
}

/// A whole AST attempt holds this gate while awaiting the engine. Actor
/// reentrancy therefore cannot interleave two environments on one interpreter.
private actor WorkflowInterpreterSerialExecutor {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<WorkflowCancellationReason?, Never>
        let cancellation: WorkflowCancellationToken
        var observerID: UUID?
    }

    private var occupied = false
    private var waiters: [Waiter] = []

    func executeAttempt(
        program: WorkflowCheckedProgram,
        limits: WorkflowInterpreterLimits,
        engine: any WorkflowCommandSink,
        context: WorkflowAttemptContext
    ) async throws {
        try await acquire(context.cancellation)
        do {
            let attempt = WorkflowInterpreterAttempt(
                program: program,
                limits: limits,
                engine: engine,
                context: context)
            try await attempt.run()
            release()
        } catch {
            release()
            throw error
        }
    }

    private func acquire(_ cancellation: WorkflowCancellationToken) async throws {
        if Task.isCancelled {
            cancellation.cancel(.requested)
        }
        guard !cancellation.isCancelled else {
            throw CancellationError()
        }
        guard occupied else {
            occupied = true
            return
        }
        let reason = await withCheckedContinuation { continuation in
            let waiterID = UUID()
            waiters.append(Waiter(
                id: waiterID,
                continuation: continuation,
                cancellation: cancellation,
                observerID: nil))
            let observerID = cancellation.observeCancellation { [weak self] reason in
                guard let self else { return }
                Task { await self.cancelWaiter(waiterID, reason: reason) }
            }
            if let observerID, let index = waiters.firstIndex(where: { $0.id == waiterID }) {
                waiters[index].observerID = observerID
            }
        }
        if reason != nil { throw CancellationError() }
    }

    private func release() {
        guard !waiters.isEmpty else {
            occupied = false
            return
        }
        let waiter = waiters.removeFirst()
        if let observerID = waiter.observerID {
            waiter.cancellation.removeCancellationObserver(observerID)
        }
        waiter.continuation.resume(returning: nil)
    }

    private func cancelWaiter(_ id: UUID, reason: WorkflowCancellationReason) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        if let observerID = waiter.observerID {
            waiter.cancellation.removeCancellationObserver(observerID)
        }
        waiter.continuation.resume(returning: reason)
    }
}

final class WorkflowInterpreter {
    private let limits: WorkflowInterpreterLimits
    private let engine: any WorkflowCommandSink
    private let serialExecutor = WorkflowInterpreterSerialExecutor()
    private let stateLock = NSLock()
    private var activeTokens: [UUID: WorkflowCancellationToken] = [:]
    private var cancelBeforeNextAttempt = false

    init(limits: WorkflowInterpreterLimits, engine: any WorkflowCommandSink) {
        self.limits = limits
        self.engine = engine
    }

    func execute(
        program: WorkflowCheckedProgram,
        deadline: ContinuousClock.Instant? = nil
    ) async throws {
        let token = WorkflowCancellationToken()
        register(token)
        if Task.isCancelled {
            token.cancel(.requested)
        }
        let deadlineMonitor = makeDeadlineMonitor(deadline: deadline, token: token)
        let context = WorkflowAttemptContext(cancellation: token, deadline: deadline)

        do {
            try await withTaskCancellationHandler {
                try await serialExecutor.executeAttempt(
                    program: program,
                    limits: limits,
                    engine: engine,
                    context: context)
            } onCancel: {
                token.cancel(.requested)
            }
        } catch {
            deadlineMonitor?.cancel()
            unregister(token)
            throw normalized(error, token: token)
        }

        deadlineMonitor?.cancel()
        unregister(token)
    }

    /// Cancels all active or queued attempts. A request made between attempts
    /// applies to the next attempt instead of becoming a stale cancellation.
    func requestCancel() {
        stateLock.lock()
        if activeTokens.isEmpty {
            cancelBeforeNextAttempt = true
        }
        let tokens = Array(activeTokens.values)
        stateLock.unlock()

        for token in tokens {
            token.cancel(.requested)
        }
    }

    private func register(_ token: WorkflowCancellationToken) {
        stateLock.lock()
        if cancelBeforeNextAttempt {
            token.cancel(.requested)
            cancelBeforeNextAttempt = false
        }
        activeTokens[UUID()] = token
        stateLock.unlock()
    }

    private func unregister(_ token: WorkflowCancellationToken) {
        stateLock.lock()
        if let entry = activeTokens.first(where: { $0.value === token }) {
            activeTokens.removeValue(forKey: entry.key)
        }
        stateLock.unlock()
    }

    private func makeDeadlineMonitor(
        deadline: ContinuousClock.Instant?,
        token: WorkflowCancellationToken
    ) -> Task<Void, Never>? {
        guard let deadline else { return nil }
        let clock = ContinuousClock()
        guard deadline > clock.now else {
            token.cancel(.deadlineExceeded)
            return nil
        }
        return Task {
            do {
                try await clock.sleep(until: deadline)
                token.cancel(.deadlineExceeded)
            } catch {
                // The monitor is cancelled after the attempt resolves.
            }
        }
    }

    private func normalized(_ error: Error, token: WorkflowCancellationToken) -> Error {
        switch token.cancellationReason {
        case .deadlineExceeded:
            return WorkflowError(
                kind: .resourceLimit,
                message: "Workflow attempt deadline exceeded.",
                site: nil)
        case .requested:
            return WorkflowError(
                kind: .cancelled,
                message: "Workflow attempt was cancelled.",
                site: nil)
        case nil:
            if error is CancellationError {
                return WorkflowError(
                    kind: .cancelled,
                    message: "Workflow attempt was cancelled.",
                    site: nil)
            }
            return error
        }
    }
}

private final class WorkflowInterpreterAttempt {
    private let program: WorkflowScriptAST
    private let limits: WorkflowInterpreterLimits
    private let engine: any WorkflowCommandSink
    private let context: WorkflowAttemptContext
    private let frozenArguments: WorkflowCanonicalValue
    private let sites: [WorkflowInterpreterSiteAddress: WorkflowStaticSite]
    private var siteSequence = WorkflowSiteKeySequence()
    private var executedNodes = 0
    private var facadeOperations = 0
    private var declaredActors: Set<String> = []
    private var commandRules: [String: [String: WorkflowCommandArgumentRule]] = [:]

    init(
        program: WorkflowCheckedProgram,
        limits: WorkflowInterpreterLimits,
        engine: any WorkflowCommandSink,
        context: WorkflowAttemptContext
    ) {
        self.program = program.ast
        self.limits = limits
        self.engine = engine
        self.context = context
        frozenArguments = .object(program.descriptor.args.allValues.mapValues(WorkflowCanonicalValue.string))
        sites = WorkflowInterpreterSiteMap.make(for: program.ast)
    }

    func run() async throws {
        guard program.version == WorkflowScriptAST.currentVersion,
              program.facadeVersion == WorkflowFacade.version
        else {
            throw failure(.authoring, "Workflow AST or facade version is unsupported.")
        }
        guard limits.maximumExecutedNodes >= 0,
              limits.maximumFacadeOperations >= 0,
              limits.maximumActors >= 0,
              limits.maximumLoopItems >= 0
        else {
            throw failure(.resourceLimit, "Workflow interpreter limits cannot be negative.")
        }

        try await executeBlock(program.body, in: WorkflowInterpreterEnvironment())
        try checkStopped(site: nil)
    }

    private func executeBlock(
        _ block: WorkflowBlock,
        in environment: WorkflowInterpreterEnvironment
    ) async throws {
        for statement in block.statements {
            try chargeNode(site: nil)
            try await execute(statement, in: environment)
        }
    }

    private func execute(
        _ statement: WorkflowStatement,
        in environment: WorkflowInterpreterEnvironment
    ) async throws {
        switch statement.kind {
        case .declaration(let name, _, let value):
            guard name != "args" else {
                throw failure(.sandboxRefusal, "The frozen args facade cannot be shadowed.")
            }
            guard !environment.containsLocal(name) else {
                throw failure(.authoring, "Workflow const binding '\(name)' is already declared.")
            }
            let resolved = try await evaluate(value, in: environment)
            environment.define(name, value: resolved)
        case .expression(let expression):
            try await executeExpressionStatement(expression, in: environment)
        case .conditional(let condition, let thenBlock, let elseBlock):
            let resolved = try await evaluate(condition, in: environment)
            if truthy(resolved) {
                try await executeBlock(thenBlock, in: WorkflowInterpreterEnvironment(parent: environment))
            } else if let elseBlock {
                try await executeBlock(elseBlock, in: WorkflowInterpreterEnvironment(parent: environment))
            }
        case .forOf(let name, _, let sequence, let body):
            guard name != "args" else {
                throw failure(.sandboxRefusal, "A loop binding cannot shadow the frozen args facade.")
            }
            let resolved = try await evaluate(sequence, in: environment)
            guard case .array(let items) = resolved else {
                throw failure(.validation, "A workflow for-of sequence must resolve to an array.")
            }
            let maximumItems = min(100, limits.maximumLoopItems)
            guard items.count <= maximumItems else {
                throw failure(.resourceLimit, "Workflow loop exceeds the \(maximumItems)-item limit.")
            }
            for item in items {
                try chargeNode(site: nil)
                let iterationEnvironment = WorkflowInterpreterEnvironment(parent: environment)
                iterationEnvironment.define(name, value: item)
                try await executeBlock(body, in: iterationEnvironment)
            }
        }
    }

    private func executeExpressionStatement(
        _ expression: WorkflowExpression,
        in environment: WorkflowInterpreterEnvironment
    ) async throws {
        try chargeNode(site: nil)
        switch expression.kind {
        case .call(let call) where call.target == .agent:
            try validateActorDeclaration(call)
        case .call(let call) where call.target == .command:
            try validateCommandDeclaration(call)
        case .call(let call) where call.target == .phase:
            let name = try literalString(call.arguments, index: 0, target: .phase)
            _ = try await invoke(.phase(name: name), at: nil)
        case .awaited(let call) where call.target == .report || call.target == .artifact:
            _ = try await executeAwaited(call, in: environment)
        default:
            throw failure(.sandboxRefusal, "This expression is outside the workflow facade.")
        }
    }

    private func validateActorDeclaration(_ call: WorkflowCall) throws {
        guard call.arguments.count == 2,
              case .literal(.string(let name)) = call.arguments[0].kind,
              case .literal(.string) = call.arguments[1].kind
        else {
            throw failure(.sandboxRefusal, "Actor declarations must contain two fixed strings.")
        }
        guard declaredActors.count < limits.maximumActors else {
            throw failure(.resourceLimit, "Workflow actor limit exceeded.")
        }
        guard declaredActors.insert(name).inserted else {
            throw failure(.authoring, "Workflow actor '\(name)' is declared more than once.")
        }
    }

    private func validateCommandDeclaration(_ call: WorkflowCall) throws {
        guard call.arguments.count == 2,
              case .literal(.string(let key)) = call.arguments[0].kind,
              !key.isEmpty,
              case .object(let members) = call.arguments[1].kind
        else {
            throw failure(.sandboxRefusal, "Command declarations must be statically defined.")
        }
        let fields = Dictionary(members.map { ($0.name, $0.value) }, uniquingKeysWith: { _, latest in latest })
        guard Set(fields.keys) == Set(["executable", "workingDirectory", "argv"]),
              fields.values.allSatisfy(isStaticExpression),
              let executable = fields["executable"],
              case .literal(.string(let path)) = executable.kind,
              path.hasPrefix("/"),
              let workingDirectory = fields["workingDirectory"],
              case .literal(.string) = workingDirectory.kind,
              let argv = fields["argv"],
              case .array(let argvValues) = argv.kind
        else {
            throw failure(.sandboxRefusal, "Command paths and argv must remain statically pinned.")
        }

        guard commandRules[key] == nil else {
            throw failure(.authoring, "Command key '\(key)' is declared more than once.")
        }
        var rules: [String: WorkflowCommandArgumentRule] = [:]
        for argument in argvValues {
            switch argument.kind {
            case .literal(.string):
                break
            case .object:
                let rule = try commandArgumentRule(argument)
                guard rules[rule.name] == nil else {
                    throw failure(.authoring, "Command argument slot '\(rule.name)' is declared more than once.")
                }
                rules[rule.name] = rule
            default:
                throw failure(.sandboxRefusal, "Command argv entries must be fixed strings or static slots.")
            }
        }
        commandRules[key] = rules
    }

    private func commandArgumentRule(_ expression: WorkflowExpression) throws -> WorkflowCommandArgumentRule {
        guard case .object(let members) = expression.kind else {
            throw failure(.sandboxRefusal, "Command argument slots must be static objects.")
        }
        let fields = Dictionary(members.map { ($0.name, $0.value) }, uniquingKeysWith: { _, latest in latest })
        guard let nameExpression = fields["name"],
              case .literal(.string(let name)) = nameExpression.kind,
              !name.isEmpty,
              let kindExpression = fields["kind"],
              case .literal(.string(let kindName)) = kindExpression.kind
        else {
            throw failure(.sandboxRefusal, "Command argument slot names and kinds must be literals.")
        }

        switch kindName {
        case WorkflowCommandArgumentKind.workspaceInputPath.rawValue:
            guard Set(fields.keys) == Set(["name", "kind"]) else {
                throw failure(.sandboxRefusal, "workspaceInputPath slots accept only name and kind.")
            }
            return WorkflowCommandArgumentRule(
                name: name,
                kind: .workspaceInputPath,
                allowedValues: nil,
                maximumBytes: 4_096)
        case WorkflowCommandArgumentKind.allowedValue.rawValue:
            guard Set(fields.keys) == Set(["name", "kind", "values"]),
                  let valuesExpression = fields["values"],
                  case .array(let valueExpressions) = valuesExpression.kind
            else {
                throw failure(.sandboxRefusal, "allowedValue slots require a static values array.")
            }
            var values: [String] = []
            for valueExpression in valueExpressions {
                guard case .literal(.string(let value)) = valueExpression.kind else {
                    throw failure(.sandboxRefusal, "Allowed command values must be string literals.")
                }
                values.append(value)
            }
            return WorkflowCommandArgumentRule(
                name: name,
                kind: .allowedValue,
                allowedValues: values,
                maximumBytes: values.map { $0.utf8.count }.max() ?? 0)
        case WorkflowCommandArgumentKind.boundedText.rawValue:
            guard Set(fields.keys) == Set(["name", "kind", "maximumBytes"]),
                  let maximumBytesExpression = fields["maximumBytes"],
                  case .literal(.number(let maximumBytesValue)) = maximumBytesExpression.kind,
                  maximumBytesValue.isFinite,
                  maximumBytesValue.rounded(.towardZero) == maximumBytesValue,
                  maximumBytesValue >= 1,
                  maximumBytesValue <= 4_096
            else {
                throw failure(.sandboxRefusal, "boundedText slots require a static byte limit from 1 to 4096.")
            }
            return WorkflowCommandArgumentRule(
                name: name,
                kind: .boundedText,
                allowedValues: nil,
                maximumBytes: Int(maximumBytesValue))
        default:
            throw failure(.sandboxRefusal, "Command argument slot kind is not supported.")
        }
    }

    private func evaluate(
        _ expression: WorkflowExpression,
        in environment: WorkflowInterpreterEnvironment
    ) async throws -> WorkflowCanonicalValue {
        try chargeNode(site: nil)
        switch expression.kind {
        case .literal(let literal):
            switch literal {
            case .null:
                return .null
            case .boolean(let value):
                return .boolean(value)
            case .number(let value):
                guard value.isFinite else {
                    throw failure(.validation, "Workflow numbers must be finite.")
                }
                return .number(value)
            case .string(let value):
                return .string(value)
            }
        case .identifier(let name):
            if name == "args" { return frozenArguments }
            guard let value = environment.value(named: name) else {
                throw failure(.sandboxRefusal, "Unknown workflow identifier '\(name)' is outside the facade.")
            }
            return value
        case .member(let base, let name):
            let value = try await evaluate(base, in: environment)
            switch value {
            case .object(let fields):
                return fields[name] ?? .null
            case .array(let items) where name == "length":
                return .integer(Int64(items.count))
            case .string(let string) where name == "length":
                return .integer(Int64(string.utf16.count))
            default:
                return .null
            }
        case .array(let expressions):
            var items: [WorkflowCanonicalValue] = []
            items.reserveCapacity(expressions.count)
            for item in expressions {
                items.append(try await evaluate(item, in: environment))
            }
            return .array(items)
        case .object(let members):
            var fields: [String: WorkflowCanonicalValue] = [:]
            for member in members {
                fields[member.name] = try await evaluate(member.value, in: environment)
            }
            return .object(fields)
        case .unaryNot(let operand):
            return .boolean(!truthy(try await evaluate(operand, in: environment)))
        case .binary(let left, let op, let right):
            return try await evaluateBinary(left, op: op, rightExpression: right, in: environment)
        case .awaited(let call):
            return try await executeAwaited(call, in: environment)
        case .call(let call):
            guard call.target == .parallel, call.arguments.count == 1 else {
                throw failure(.sandboxRefusal, "Only the parallel graph constructor is synchronous.")
            }
            let nodes = try await evaluate(call.arguments[0], in: environment)
            guard case .array(let graphNodes) = nodes else {
                throw failure(.validation, "parallel requires an array of graph nodes.")
            }
            guard graphNodes.count <= 100 else {
                throw failure(.resourceLimit, "Workflow graphs are limited to 100 nodes.")
            }
            return .object(["$workflowGraph": .array(graphNodes)])
        }
    }

    private func evaluateBinary(
        _ leftExpression: WorkflowExpression,
        op: WorkflowBinaryOperator,
        rightExpression: WorkflowExpression,
        in environment: WorkflowInterpreterEnvironment
    ) async throws -> WorkflowCanonicalValue {
        let left = try await evaluate(leftExpression, in: environment)
        if op == .logicalAnd, !truthy(left) { return left }
        if op == .logicalOr, truthy(left) { return left }

        let right = try await evaluate(rightExpression, in: environment)
        switch op {
        case .logicalAnd:
            return right
        case .logicalOr:
            return right
        case .strictEqual:
            return .boolean(strictlyEqual(left, right))
        case .strictNotEqual:
            return .boolean(!strictlyEqual(left, right))
        case .lessThan, .lessThanOrEqual, .greaterThan, .greaterThanOrEqual:
            return .boolean(try compare(left, op: op, right))
        }
    }

    private func strictlyEqual(_ left: WorkflowCanonicalValue, _ right: WorkflowCanonicalValue) -> Bool {
        switch (left, right) {
        case (.integer(let integer), .number(let number)),
             (.number(let number), .integer(let integer)):
            // Source literals and native observation counts share one numeric type in the facade.
            return Int64(exactly: number) == integer
        default:
            return left == right
        }
    }

    private func compare(
        _ left: WorkflowCanonicalValue,
        op: WorkflowBinaryOperator,
        _ right: WorkflowCanonicalValue
    ) throws -> Bool {
        let ordering: ComparisonResult
        switch (left, right) {
        case (.string(let lhs), .string(let rhs)):
            ordering = lhs == rhs ? .orderedSame : (lhs < rhs ? .orderedAscending : .orderedDescending)
        case (.integer(let lhs), .integer(let rhs)):
            ordering = lhs == rhs ? .orderedSame : (lhs < rhs ? .orderedAscending : .orderedDescending)
        case (.number(let lhs), .number(let rhs)):
            guard lhs.isFinite, rhs.isFinite else {
                throw failure(.validation, "Ordered comparison requires finite numbers.")
            }
            ordering = lhs == rhs ? .orderedSame : (lhs < rhs ? .orderedAscending : .orderedDescending)
        case (.integer(let lhs), .number(let rhs)):
            guard rhs.isFinite else { throw failure(.validation, "Ordered comparison requires finite numbers.") }
            ordering = Double(lhs) == rhs ? .orderedSame : (Double(lhs) < rhs ? .orderedAscending : .orderedDescending)
        case (.number(let lhs), .integer(let rhs)):
            guard lhs.isFinite else { throw failure(.validation, "Ordered comparison requires finite numbers.") }
            ordering = lhs == Double(rhs) ? .orderedSame : (lhs < Double(rhs) ? .orderedAscending : .orderedDescending)
        default:
            throw failure(.validation, "Ordered comparison requires values of the same scalar type.")
        }

        switch op {
        case .lessThan: return ordering == .orderedAscending
        case .lessThanOrEqual: return ordering != .orderedDescending
        case .greaterThan: return ordering == .orderedDescending
        case .greaterThanOrEqual: return ordering != .orderedAscending
        case .logicalAnd, .logicalOr, .strictEqual, .strictNotEqual:
            throw failure(.authoring, "Invalid ordered comparison operator.")
        }
    }

    private func executeAwaited(
        _ call: WorkflowCall,
        in environment: WorkflowInterpreterEnvironment
    ) async throws -> WorkflowCanonicalValue {
        switch call.target {
        case .ask:
            guard call.arguments.count == 3 else {
                throw failure(.sandboxRefusal, "ask requires an actor, prompt, and static shape.")
            }
            let actor = try literalString(call.arguments, index: 0, target: .ask)
            let prompt = try await evaluate(call.arguments[1], in: environment)
            let shape = try await evaluate(call.arguments[2], in: environment)
            let site = try site(for: call, lane: actor)
            return try await invoke(.ask(actor: actor, prompt: prompt, shape: shape), at: site)
        case .join:
            guard call.arguments.count == 1 else {
                throw failure(.sandboxRefusal, "join requires one graph handle.")
            }
            let graph = try await evaluate(call.arguments[0], in: environment)
            guard case .object(let fields) = graph, fields["$workflowGraph"] != nil else {
                throw failure(.validation, "join requires a graph returned by parallel.")
            }
            return try await invoke(.join(graph: graph), at: try site(for: call, lane: "main"))
        case .criticLoop:
            guard call.arguments.count == 1 else {
                throw failure(.sandboxRefusal, "criticLoop requires one static policy.")
            }
            let policy = try await evaluate(call.arguments[0], in: environment)
            return try await invoke(.criticLoop(policy: policy), at: try site(for: call, lane: "main"))
        case .worldRead:
            let operation = try staticWorldReadOperation(call)
            return try await invoke(.worldRead(operation: operation), at: try site(for: call, lane: "main"))
        case .run:
            guard call.arguments.count == 2,
                  case .literal(.string(let commandKey)) = call.arguments[0].kind,
                  !commandKey.isEmpty
            else {
                throw failure(.sandboxRefusal, "run requires a literal key for a pinned command.")
            }
            let resolvedValues = try await evaluate(call.arguments[1], in: environment)
            guard case .object(let values) = resolvedValues else {
                throw failure(.validation, "run requires named dynamic command values.")
            }
            let reservedFields: Set<String> = ["executable", "workingDirectory", "argv"]
            guard values.keys.allSatisfy({ !reservedFields.contains($0) && !$0.isEmpty }) else {
                throw failure(.sandboxRefusal, "run cannot supply executable, workingDirectory, or argv.")
            }
            guard let rules = commandRules[commandKey] else {
                throw failure(.validation, "run key '\(commandKey)' has no earlier command declaration.")
            }
            guard Set(values.keys) == Set(rules.keys) else {
                throw failure(.validation, "run values must match the declared dynamic command slots.")
            }
            for (name, rule) in rules {
                guard case .string(let value)? = values[name] else {
                    throw failure(.validation, "Dynamic command value '\(name)' must resolve to a string.")
                }
                guard value.utf8.count <= rule.maximumBytes else {
                    throw failure(.validation, "Dynamic command value '\(name)' exceeds its byte limit.")
                }
                switch rule.kind {
                case .allowedValue:
                    guard rule.allowedValues?.contains(value) == true else {
                        throw failure(.validation, "Dynamic command value '\(name)' is not allowed.")
                    }
                case .boundedText:
                    break
                case .workspaceInputPath:
                    let components = value.split(separator: "/", omittingEmptySubsequences: false)
                    guard !value.isEmpty,
                          !value.hasPrefix("/"),
                          !value.utf8.contains(0),
                          components.allSatisfy({ $0 != ".." })
                    else {
                        throw failure(.validation, "Dynamic path '\(name)' must stay workspace-relative.")
                    }
                }
            }
            return try await invoke(
                .run(commandKey: commandKey, values: values),
                at: try site(for: call, lane: "main"))
        case .report, .artifact:
            guard call.arguments.count == 1 else {
                throw failure(.sandboxRefusal, "report and artifact require one value.")
            }
            let value = try await evaluate(call.arguments[0], in: environment)
            do {
                _ = try WorkflowCanonicalSerialization.encodedData(value)
            } catch {
                throw failure(.validation, "Published workflow values must be canonical and finite.")
            }
            let operation: WorkflowInterpreterOperation = call.target == .report
                ? .report(value: value)
                : .artifact(value: value)
            return try await invoke(operation, at: try site(for: call, lane: "main"))
        default:
            throw failure(.sandboxRefusal, "Awaited call is outside the workflow facade.")
        }
    }

    private func staticWorldReadOperation(_ call: WorkflowCall) throws -> WorkflowCanonicalValue {
        guard call.arguments.count == 1,
              case .call(let selector) = call.arguments[0].kind
        else {
            throw failure(.sandboxRefusal, "world.read requires a fixed read selector.")
        }

        let expectedCount: Int
        switch selector.target {
        case .glob, .git: expectedCount = 1
        case .read: expectedCount = 2
        case .grep: expectedCount = selector.arguments.count == 1 ? 1 : 2
        default:
            throw failure(.sandboxRefusal, "world.read selector is outside the read-only facade.")
        }
        guard selector.arguments.count == expectedCount,
              selector.arguments.allSatisfy(isStaticExpression)
        else {
            throw failure(.sandboxRefusal, "world.read selector fields must be static literals.")
        }
        let values = try selector.arguments.map(staticValue)
        switch selector.target {
        case .glob, .grep:
            guard values.first.map(isString) == true else {
                throw failure(.validation, "World read patterns must be strings.")
            }
        case .read:
            guard values.count == 2, isString(values[0]), isPositiveInteger(values[1]) else {
                throw failure(.validation, "World read paths and byte limits are invalid.")
            }
        case .git:
            guard case .string(let operation) = values[0],
                  ["status", "diff", "log", "changedFiles"].contains(operation)
            else {
                throw failure(.sandboxRefusal, "git read operation is not supported.")
            }
        default:
            throw failure(.sandboxRefusal, "world.read selector is outside the read-only facade.")
        }

        return .object([
            "kind": .string(selector.target.rawValue),
            "arguments": .array(values)
        ])
    }

    private func invoke(
        _ operation: WorkflowInterpreterOperation,
        at site: WorkflowSiteKey?
    ) async throws -> WorkflowCanonicalValue {
        try checkStopped(site: site)
        guard facadeOperations < limits.maximumFacadeOperations else {
            throw failure(.resourceLimit, "Workflow facade-operation budget exceeded.", site: site)
        }
        facadeOperations += 1
        do {
            let value = try await engine.perform(operation, at: site, context: context)
            try checkStopped(site: site)
            return value
        } catch {
            if let reason = context.cancellation.cancellationReason {
                throw stoppedError(reason, site: site)
            }
            throw error
        }
    }

    private func site(for call: WorkflowCall, lane: String) throws -> WorkflowSiteKey {
        let address = WorkflowInterpreterSiteAddress(
            lane: lane,
            sourceOffset: call.sourceRange.start.byteOffset)
        guard let staticSite = sites[address] else {
            throw failure(.authoring, "Facade call has no stable checked site.")
        }
        do {
            return try siteSequence.next(for: staticSite)
        } catch {
            throw failure(.resourceLimit, "Facade call site ordinal exhausted.")
        }
    }

    private func chargeNode(site: WorkflowSiteKey?) throws {
        try checkStopped(site: site)
        guard executedNodes < limits.maximumExecutedNodes else {
            throw failure(.resourceLimit, "Workflow executed-node budget exceeded.", site: site)
        }
        executedNodes += 1
    }

    private func checkStopped(site: WorkflowSiteKey?) throws {
        if Task.isCancelled {
            context.cancellation.cancel(.requested)
        }
        if let deadline = context.deadline, deadline <= ContinuousClock().now {
            context.cancellation.cancel(.deadlineExceeded)
        }
        if let reason = context.cancellation.cancellationReason {
            throw stoppedError(reason, site: site)
        }
    }

    private func stoppedError(
        _ reason: WorkflowCancellationReason,
        site: WorkflowSiteKey?
    ) -> WorkflowError {
        switch reason {
        case .requested:
            return WorkflowError(kind: .cancelled, message: "Workflow attempt was cancelled.", site: site)
        case .deadlineExceeded:
            return WorkflowError(
                kind: .resourceLimit,
                message: "Workflow attempt deadline exceeded.",
                site: site)
        }
    }

    private func failure(
        _ kind: WorkflowErrorKind,
        _ message: String,
        site: WorkflowSiteKey? = nil
    ) -> WorkflowError {
        WorkflowError(kind: kind, message: message, site: site)
    }

    private func literalString(
        _ arguments: [WorkflowExpression],
        index: Int,
        target: WorkflowCallTarget
    ) throws -> String {
        guard arguments.indices.contains(index),
              case .literal(.string(let value)) = arguments[index].kind
        else {
            throw failure(.sandboxRefusal, "\(target.rawValue) requires a fixed string argument.")
        }
        return value
    }

    private func staticValue(_ expression: WorkflowExpression) throws -> WorkflowCanonicalValue {
        switch expression.kind {
        case .literal(let literal):
            switch literal {
            case .null: return .null
            case .boolean(let value): return .boolean(value)
            case .number(let value):
                guard value.isFinite else { throw failure(.validation, "Static number must be finite.") }
                return .number(value)
            case .string(let value): return .string(value)
            }
        case .array(let expressions):
            return .array(try expressions.map(staticValue))
        case .object(let members):
            return .object(try Dictionary(
                members.map { ($0.name, try staticValue($0.value)) },
                uniquingKeysWith: { _, latest in latest }))
        default:
            throw failure(.sandboxRefusal, "This facade position accepts literal values only.")
        }
    }

    private func isStaticExpression(_ expression: WorkflowExpression) -> Bool {
        switch expression.kind {
        case .literal:
            return true
        case .array(let items):
            return items.allSatisfy(isStaticExpression)
        case .object(let members):
            return members.allSatisfy { isStaticExpression($0.value) }
        default:
            return false
        }
    }

    private func isString(_ value: WorkflowCanonicalValue) -> Bool {
        if case .string = value { return true }
        return false
    }

    private func isPositiveInteger(_ value: WorkflowCanonicalValue) -> Bool {
        switch value {
        case .integer(let integer): return integer > 0
        case .number(let number): return number.isFinite && number > 0 && number.rounded(.towardZero) == number
        default: return false
        }
    }

    private func truthy(_ value: WorkflowCanonicalValue) -> Bool {
        switch value {
        case .null:
            return false
        case .boolean(let value):
            return value
        case .integer(let value):
            return value != 0
        case .number(let value):
            return value != 0 && !value.isNaN
        case .string(let value):
            return !value.isEmpty
        case .array, .object:
            return true
        }
    }
}

private final class WorkflowInterpreterEnvironment {
    private let parent: WorkflowInterpreterEnvironment?
    private var values: [String: WorkflowCanonicalValue] = [:]

    init(parent: WorkflowInterpreterEnvironment? = nil) {
        self.parent = parent
    }

    func containsLocal(_ name: String) -> Bool {
        values[name] != nil
    }

    func define(_ name: String, value: WorkflowCanonicalValue) {
        values[name] = value
    }

    func value(named name: String) -> WorkflowCanonicalValue? {
        values[name] ?? parent?.value(named: name)
    }
}

private struct WorkflowInterpreterSiteAddress: Hashable {
    let lane: String
    let sourceOffset: Int
}

private enum WorkflowInterpreterSiteMap {
    private struct Candidate {
        let lane: String
        let sourceOffset: Int
        let insertionOrder: Int
    }

    static func make(for program: WorkflowScriptAST) -> [WorkflowInterpreterSiteAddress: WorkflowStaticSite] {
        var collector = Collector()
        collector.collect(program.body, constants: [:])
        var nextIndexByLane: [String: Int] = [:]
        var result: [WorkflowInterpreterSiteAddress: WorkflowStaticSite] = [:]
        for candidate in collector.candidates.sorted(by: {
            if $0.sourceOffset != $1.sourceOffset { return $0.sourceOffset < $1.sourceOffset }
            return $0.insertionOrder < $1.insertionOrder
        }) {
            let index = nextIndexByLane[candidate.lane, default: 0]
            nextIndexByLane[candidate.lane] = index + 1
            result[WorkflowInterpreterSiteAddress(
                lane: candidate.lane,
                sourceOffset: candidate.sourceOffset)] = WorkflowStaticSite(
                    lane: candidate.lane,
                    siteIndex: index)
        }
        return result
    }

    private struct Collector {
        private(set) var candidates: [Candidate] = []
        private var insertionOrder = 0

        mutating func collect(_ block: WorkflowBlock, constants initial: [String: WorkflowExpression]) {
            var constants = initial
            for statement in block.statements {
                switch statement.kind {
                case .declaration(let name, _, let expression):
                    collect(expression, constants: constants)
                    constants[name] = expression
                case .expression(let expression):
                    collect(expression, constants: constants)
                case .conditional(let condition, let thenBlock, let elseBlock):
                    collect(condition, constants: constants)
                    collect(thenBlock, constants: constants)
                    if let elseBlock { collect(elseBlock, constants: constants) }
                case .forOf(let name, _, let sequence, let body):
                    collect(sequence, constants: constants)
                    var loopConstants = constants
                    loopConstants.removeValue(forKey: name)
                    collect(body, constants: loopConstants)
                }
            }
        }

        private mutating func collect(_ expression: WorkflowExpression, constants: [String: WorkflowExpression]) {
            switch expression.kind {
            case .call(let call), .awaited(let call):
                collect(call, constants: constants)
                for argument in call.arguments { collect(argument, constants: constants) }
            case .member(let base, _), .unaryNot(let base):
                collect(base, constants: constants)
            case .array(let values):
                for value in values { collect(value, constants: constants) }
            case .object(let members):
                for member in members { collect(member.value, constants: constants) }
            case .binary(let left, _, let right):
                collect(left, constants: constants)
                collect(right, constants: constants)
            case .literal, .identifier:
                break
            }
        }

        private mutating func collect(_ call: WorkflowCall, constants: [String: WorkflowExpression]) {
            switch call.target {
            case .ask:
                if let actor = stringLiteral(at: 0, in: call) {
                    add(lane: actor, offset: call.sourceRange.start.byteOffset)
                }
            case .parallel:
                add(lane: "main", offset: call.sourceRange.start.byteOffset)
                if let nodes = call.arguments.first.flatMap({ resolve($0, constants: constants) }),
                   case .array(let values) = nodes.kind
                {
                    for node in values {
                        guard case .object(let members) = node.kind,
                              let actorExpression = memberMap(members)["actor"],
                              case .literal(.string(let actor)) = actorExpression.kind
                        else { continue }
                        add(lane: actor, offset: actorExpression.sourceRange.start.byteOffset)
                    }
                }
            case .criticLoop:
                add(lane: "main", offset: call.sourceRange.start.byteOffset)
                if let policy = call.arguments.first.flatMap({ resolve($0, constants: constants) }),
                   case .object(let members) = policy.kind
                {
                    let policyFields = memberMap(members)
                    for laneName in ["producer", "critic"] {
                        guard let laneExpression = policyFields[laneName],
                              let lane = resolve(laneExpression, constants: constants),
                              case .object(let laneMembers) = lane.kind,
                              let actorExpression = memberMap(laneMembers)["actor"],
                              case .literal(.string(let actor)) = actorExpression.kind
                        else { continue }
                        add(lane: actor, offset: actorExpression.sourceRange.start.byteOffset)
                    }
                }
            case .join, .worldRead, .run, .report, .artifact:
                add(lane: "main", offset: call.sourceRange.start.byteOffset)
            case .agent, .phase, .command, .glob, .read, .grep, .git:
                break
            }
        }

        private mutating func add(lane: String, offset: Int) {
            candidates.append(Candidate(lane: lane, sourceOffset: offset, insertionOrder: insertionOrder))
            insertionOrder += 1
        }

        private func resolve(
            _ expression: WorkflowExpression,
            constants: [String: WorkflowExpression]
        ) -> WorkflowExpression? {
            var current = expression
            var visited: Set<String> = []
            while case .identifier(let name) = current.kind {
                guard visited.insert(name).inserted, let value = constants[name] else { return nil }
                current = value
            }
            return current
        }

        private func memberMap(_ members: [WorkflowObjectMember]) -> [String: WorkflowExpression] {
            Dictionary(members.map { ($0.name, $0.value) }, uniquingKeysWith: { _, latest in latest })
        }

        private func stringLiteral(at index: Int, in call: WorkflowCall) -> String? {
            guard call.arguments.indices.contains(index),
                  case .literal(.string(let value)) = call.arguments[index].kind
            else { return nil }
            return value
        }
    }
}
