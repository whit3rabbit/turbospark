# Swift context references (the typed `@` schemes)

What a typed `@...` token can mean in the composer, how it expands at send
time, and the security rules it must not break. The grammar is
`Files/ContextReferenceParser.swift` (pure), the resolution is
`Files/MentionResolver.swift` plus `Files/ContextReferenceContent.swift`
(git and web), and the popup rows are
`Generation/ComposerAutocompleteEngine.swift`.

## The surface

Typing `@` in the composer opens the autocomplete popup. On an empty query
it offers the six reference rows first, then the project tree browse; a
typed prefix filters both. Accepting a COMPLETE row (`@diff`, `@staged`,
`@file:src/App.swift`) inserts it with a trailing space and closes the
popup; accepting a PREFIX row (`@file:`, `@git:`, `@url:`, `@folder:`)
inserts it WITHOUT a space and the same text change re-opens the popup for
what follows the colon (`ComposerAutocompleteController.accept`).

## The grammar

Tokens come from the one mention grammar: start-of-text or whitespace
before the `@`, no match mid-word (an email address is never a mention),
`@@` is an escaped literal, and `@"quoted path"` carries paths with spaces.
Deduplication is by token body. Each token then parses to one reference:

| Token | Meaning | Content source |
|---|---|---|
| `@diff` | Unstaged working tree changes | `git diff` |
| `@staged` | Staged changes | `git diff --cached` |
| `@git:N` | Last N commits with patches, N clamped to 1...10 | `git log -p -n N` |
| `@url:...` | A web page, fetched to markdown | `WebFetchExecutor` |
| `@file:p[:N[-M]]` | File contents, optionally lines N through M | document extraction |
| `@folder:p` | A folder's supported files (bounded walk) | document extraction |
| `@path` / `@"path"` | Same as `@file:` / `@folder:`, decided by what exists | document extraction |

Two Hermes fuzz rules apply to every value: trailing punctuation
(`,.;!?`) is stripped before scheme matching ("check @diff, then @staged."
names the schemes), and an invalid line range (`:0`, `:25-10`) silently
means the full file. Line ranges are 1-indexed and inclusive, and are
sliced from the EXTRACTED text after import. A range-shaped suffix that is
actually part of a real file's name (`@notes:2024` where only
`notes:2024` exists) falls back to the raw token at resolution.

## Resolution

`MentionResolver.resolveMentions` runs inside the submission (before the
attachments are read and before `UserPromptSubmit`) and again at the
steer-boundary drain (`deliverSteersAtBoundary`) -- both call sites in
`AppModel+Submission.swift` and `AppModel+Queue.swift` share the one
resolver. Every reference becomes an ordinary `AppPromptAttachment`
appended through `appendPromptAttachmentDuringSubmission`, so the content
rides the existing pipeline: inlined into the outgoing message as
`--- Attachment:` blocks, counted by the context ring's attachments piece,
and summarized by compaction like any other attachment.

The token STAYS in the message text; the attachment carries the content.
`@diff` builds "Working tree diff", `@staged` builds "Staged diff",
`@git:N` builds "Last N commits" (all capped by the 10,000-line diff
limit the `/diff` sheet uses), and `@url:` builds an attachment named for
the host capped at the same 240,000 characters one document may extract.

## Reserved words and the quoted escape

`diff` and `staged` are reserved: a bare `@diff` is always the git
reference, even when a file of that name exists. The quoted form is
always a literal path and never parses as a scheme, so `@"diff"` names
the file. Quotes cannot ride behind a scheme prefix (`@file:"my
note.txt"` is not one token); a path with spaces under a scheme is typed
as the bare quoted `@"my note.txt"` instead, and the popup offers no
quoted paths under `@file:` / `@folder:` for the same reason.

## Security

- Project containment is unchanged from plain mentions:
  `PathContainment.resolvedIfContained` refuses absolute paths outside the
  project, traversal, and symlink escape, so a mention cannot turn
  repository-controlled text into an unapproved read.
- Beside containment, every path reference passes the read tool's own
  sensitive-file classifier (`ToolRiskClassifier.isSensitivePath`), so
  `@.env`, `@.ssh/id_rsa`, and `@secrets.json` are refused even when the
  project root is the home directory. This is defense in depth: either
  clause alone catches the filename cases.
- `@url:` reuses `WebFetchExecutor` unchanged: http/https only, the
  private-network/metadata-host block, domain validation, the 5 MB byte
  cap, and the textual-MIME gate. It is additionally gated on the chat's
  web-tools toggle (`webSearchEnabled`) -- the same switch that gates the
  model's own web tools -- and refuses with a toast when it is off.

## Failure behavior

Bare `@path` mentions keep the resolver's founding silence: an
unresolvable token is prose, the missing chip is the feedback, and a
toast on every casual `@word` would be noise. The typed schemes are
explicit asks, so their failures toast: no project or not a repository
(the git command's own stderr is the message), an empty diff ("nothing
staged to attach"), a failed or empty fetch, a sensitive path, or web
tools disabled. A git reference never probes `rev-parse` first; the
command's own failure is the diagnosis.

## Tests

`ContextReferenceParserTests` (pure grammar: reserved words, clamping,
URL schemes, line ranges, punctuation), `MentionResolverTests` (end to end
against real temp git repositories: diff/staged/log content, the
sensitive-path block, range slicing, the raw-name fallback, dedupe), and
the composer section of `ComposerAutocompleteTests` (reference rows,
scheme completion, the prefix-row accept rule). The sensitive-path test's
discriminating fixture is `secrets.json`, whose extension document
extraction supports -- `.env` would pass vacuously through extraction's
format allowlist even with the guard deleted.
