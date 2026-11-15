# Trajectory log (ZP-03)

A per-chat, local-only record of what the model and the tools did, for
debugging a bad turn in place. Design and decisions:
`docs/designs/zcode-parity/ZP-03-model-trajectory-viewer.md`.

Read this before changing `Diagnostics/RolloutStore.swift`,
`State/AppModel+Trajectory.swift` or the pane in `Generation/Trajectory*`.

## Contract

- **Off by default** (`MacAppSettings.trajectoryLoggingEnabled`, General
  settings). Turning it off, protecting a profile, deleting a chat or clearing
  it deletes the logs. Never sent anywhere.
- **Plaintext on disk.** `trajectories/rollout-<chat id>.jsonl` under the
  profile store root, 0600 in a 0700 folder. Not in the encrypted vault. That
  is why it is off by default and why a Ghost chat or a passphrase-protected
  profile never writes one (the store refuses too, not only the caller).
- **Redacted and clipped before writing**, in that order
  (`TrajectoryRedactor`, `RolloutStore.sanitize`). Best effort: it knows the
  shapes of credentials listed in its patterns, not every secret. Add a
  pattern with a test that fails without it.
- **Append after the fact**, one whole line per finished model request and per
  finished tool outcome. A crash can leave an unterminated last line: readers
  skip and count it, the next append closes it with a newline. A file whose
  first complete line is not a versioned JSON object is quarantined
  (`corrupt-...`), never overwritten. A line with a newer `v` is skipped and
  counted, not half read.
- **Bounded**: 4,000 chars per text field, 4 MiB per chat (oldest whole lines
  dropped), 30 day retention, a 500 entry window in the pane, whole-log search
  only on request and off the main thread.
- **Seams**: `.finished` in `executeGenerationTurn` and
  `appendToolExecutionTurn`. A new path that makes model calls outside them
  (ZP-23 hooks) calls `logTrajectoryModelCall` with its chat id.

Tests: `RolloutStoreTests` (mutation-checked: the tail boundary, the
truncated-final-line guard, the newline heal, the quarantine and the
redact-before-clip order each fail when removed).
