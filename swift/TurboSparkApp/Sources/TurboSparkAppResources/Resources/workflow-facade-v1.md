# Workflow facade v1

Version: `1`

The only accepted entry form is:

```js
async function workflow() {
  // restricted workflow statements
}
```

The checker parses the complete function body with the versioned grammar below.
The interpreter executes its checked syntax tree. Workflow source is never
evaluated as JavaScript and has no access to JavaScript globals or host APIs.

## Facade members

### `agent`

`agent(name, role)` declares an actor. Both arguments are string literals and
the declaration must be at the entry block's top level before the actor is used.

### `ask`

`await ask(actor, prompt, shape)` asks a declared actor for a value matching a
declared result shape. `actor` and `shape` are static; `prompt` may use earlier
values.

### `parallel`

`parallel` declares a bounded dependency graph from a source array literal of
node declarations and returns its handle. Node IDs, actors, shapes,
dependencies, and retry limits are static. Prompts may use earlier values. A
graph has at most 100 nodes.

### `join`

`await join(graph)` waits for a graph and returns its node results and statuses.
Every graph must be joined on every control-flow path that creates it.

### `criticLoop`

`await criticLoop(policy)` runs a bounded producer and critic sequence. Actor
names, shapes, verdict and feedback field names, and iteration limits are
static. Prompts may use earlier values. Iteration limits are literals from 1
through 10.

### `phase`

`phase(name)` marks a run phase. The name is a string literal and the call must
be at the entry block's top level.

### `world`

`await world.read(operation)` performs a declared journaled read. The operation
must be one of these read-only forms:

| Read form | Arguments |
| --- | --- |
| `glob(pattern)` | A workspace-relative glob pattern. |
| `read(path, maxBytes)` | A workspace-relative path and a literal byte limit. |
| `grep(pattern, pathHint?)` | A search pattern and an optional workspace-relative path hint. |
| `git(op)` | One literal operation: `status`, `diff`, `log`, or `changedFiles`. |

The operation kind and its selectors are static. Returned observations are
dynamic data values. Reads are bounded, journaled, read-only, and contained
under the workspace root; the world surface does not expose general process or
network access.

### `command`

`command(key, definition)` declares a pinned command. The key, absolute
executable path, working directory, argv length, fixed argv strings, dynamic
slot names and kinds, allowed-value lists, and byte limits are static. Dynamic
argv slots are limited to `workspaceInputPath`, `allowedValue`, and
`boundedText`; the bounded-text limit is a literal from 1 through 4,096 bytes.
Command declarations must be at the entry block's top level and precede use.

### `run`

`await run(commandKey, dynamicValues)` dispatches a previously declared command
by its string-literal key and supplies only named dynamic argument values. It
cannot declare, select, or replace an executable, working directory, argv
shape, or fixed argument. Runtime values are checked against the command pin
immediately before dispatch.

Accepted shape:

```js
await run("build", { sourcePath: args.sourcePath });
```

Rejected shape:

```js
await run("build", { executable: "/usr/bin/tool", argv: ["--version"] });
```

The rejected shape is included as a machine-readable conformance fixture at
the end of this reference.

### `report`

`await report(value)` publishes a serializable report value.

### `artifact`

`await artifact(value)` publishes a versioned artifact value.

### `args`

`args.name` reads a frozen value declared by a saved workflow definition. The
workflow cannot mutate its frozen arguments.

## Canonical statement productions

Each line below is an accepted statement production. The examples are
independent statements, not one combined block.

```js
agent("name", "role");
const value = await ask("actor", "prompt", shape);
const graph = parallel([{ id: "node", actor: "name", prompt: value, shape: shape, dependsOn: ["prior"], maxRetries: 1 }]);
const values = await join(group);
const review = await criticLoop(policy);
const review = await criticLoop({ producer: { actor: "writer", prompt: value, shape: shape }, critic: { actor: "reviewer", prompt: value }, verdictField: "verdict", feedbackField: "feedback", maxIterations: 3 });
phase("name");
const value = await world.read(operation);
command("key", { executable: "/absolute/path", workingDirectory: "workspace", argv: ["fixed", { name: "path", kind: "workspaceInputPath" }] });
const value = await run("commandKey", args);
await report(value);
await artifact(value);
```

Parallel nodes have the literal fields `id`, `actor`, `prompt`, `shape`,
`dependsOn`, and `maxRetries`. Critic policies have `producer` (actor, prompt,
and shape), `critic` (actor and prompt), `verdictField`, `feedbackField`, and
`maxIterations`.

## Accepted source forms

- Immutable declarations: `const name = expression;`
- Facade statements listed above.
- Branches: `if (pureExpression) { statements } else { statements }`; `else` is
  optional.
- Bounded iteration: `for (const name of expression) { statements }`. The
  expression must produce an array of at most 100 items before the body starts.
- Pure expressions: strings, finite numbers, booleans, `null`, arrays, object
  literals, earlier `const` names, `args.name`, dotted property access on data
  values, unary `!`, strict equality and inequality, boolean `&&` and `||`, and
  ordered comparisons between values of the same scalar type.

## Literal-only and dynamic-value positions

| Position | Rule |
| --- | --- |
| Actor name and role | String literals in a top-level `agent` declaration. |
| Phase name | String literal in a top-level `phase` declaration. |
| Ask actor and result shape | Declared actor name and statically declared shape. |
| Command key and definition | Static key, absolute executable string, working directory, argv length, fixed argv strings, dynamic slot names and kinds, allowed values, and byte limits. |
| Run command key | Previously declared command key as a string literal. |
| Parallel graph structure | Node IDs, actor names, shapes, dependencies, and retry limits are static. |
| Critic policy structure | Actor names, shapes, field names, and iteration limit are static. |
| World operation selectors | Read kind, glob pattern, path, byte limit, grep pattern and path hint, or git operation are static. |
| Dynamic values | Ask and graph/critic prompts, values read from `args` or earlier declarations, permitted world-operation data, named values supplied to `run`, and serializable values passed to `report` or `artifact`. |

No position permits an executable, working directory, argv length, or argument
name to be supplied dynamically. Named values passed to `run` fill slots from
the prior command declaration; they do not alter that declaration.

## Rejected syntax

The checker rejects recursion; `while` and `do` loops; numeric or
condition-based `for` loops; nested or arrow functions; mutable declarations;
compound assignment; computed property access; prototype access; `eval`;
dynamic import; `new`; exception handling; and every unlisted statement,
operator, global, or call form. `parallel` and `criticLoop` cannot appear inside
a `for...of` body.

## Run-dispatch conformance fixtures

<!-- run-dispatch-fixtures:start -->
[
  {
    "source": "await run(\"build\", { sourcePath: args.sourcePath });",
    "commandKeyForm": "stringLiteral",
    "argumentForm": "namedDynamicValues",
    "accepted": true
  },
  {
    "source": "await run(\"build\", { executable: \"/usr/bin/tool\", argv: [\"--version\"] });",
    "commandKeyForm": "stringLiteral",
    "argumentForm": "executableCommandDefinition",
    "accepted": false
  },
  {
    "source": "await run(commandName, { sourcePath: args.sourcePath });",
    "commandKeyForm": "dynamicExpression",
    "argumentForm": "namedDynamicValues",
    "accepted": false
  }
]
<!-- run-dispatch-fixtures:end -->
