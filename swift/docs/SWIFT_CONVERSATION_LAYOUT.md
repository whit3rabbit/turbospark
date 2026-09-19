# Active conversation layout

The active Swift chat uses a shared 760-point reading column with 24-point
side insets. `ConversationLayout` supplies the same measure to the transcript
and composer. The initial welcome view retains its existing layout.

`ConversationPaneView` places the composer below the scrolling transcript.
It reserves the composer's actual layout height, so expanding a draft or
showing an error cannot cover the final message. `ChatTranscriptView` owns
turn spacing, separators, navigation, and the Activity disclosure. Folding a
finished turn keeps its prompt and final answer; it does not modify history
or the model's context. The current running turn does not offer that toggle.

Long messages measure their unconstrained vertical height before clipping.
The View all / Show less control preserves the full Markdown, tables, and
HTML preview actions. Plan submissions from `exit_plan_mode` and its alias
render the supplied plan as Markdown with a bounded preview. The tool header
can fold the entire card; unsuccessful results and approvals remain visible.

Single tool calls and tool groups share header height, icon width, type role,
border, and corner radius. Expanding a group keeps its child cards aligned
with the transcript. Tool-only messages omit repeated assistant headers.
Structural surfaces use the resolved theme, including expanded tool output.
Status and risk colors retain their existing meaning.

The Chat summary button is available in every active conversation, including
chats outside a project. On a wide window the summary occupies the upper-right
column. Below the existing pinning threshold it opens as a toolbar popover.
Tasks, outputs, and sources each fold independently. Output previews and
Model Settings retain priority over the summary. No second task store exists.

Model Settings closes when a conversation starts or an existing conversation
is selected. Loading a model during that conversation does not reopen it.
The settings toolbar button and existing keyboard command can reopen it;
streamed text does not undo an explicit reopen. The toolbar button dismisses
preview panes before opening settings so the action has a visible effect.

## Verification

Run from `swift/TurboSparkApp`:

```sh
swift build
swift test
TURBOSPARK_CHAT_UI_REVIEW_DIR=/tmp/turbospark-chat-ui-review \
  swift test --filter ChatSharingAndSummaryTests/testRenderReviewArtifactsWhenRequested
```

The opt-in fixture renders the actual conversation pane in light, dark,
custom-color, narrow, large-text, and folded-turn states. It uses isolated
app stores. `testSettingsCloseWhenConversationStartsAndReopenOnRequest`
mounts `RootView` against isolated preferences and exercises first-message,
manual reopen, streaming, and chat-switch transitions. The summary eligibility
case covers both project and ordinary chats. New visible label: `Plan`,
translated into all catalog languages.

Verification on 2026-09-19: the app build and 34 focused chat, layout, and font
checks passed. Both settings visibility and ordinary-chat summary eligibility
were mutation-checked; reverting either behavior failed its corresponding
case. Six native view renders were inspected: light, dark, narrow, custom
colors with reduced transparency, larger text, and folded activity.

The full app suite ran 1,798 tests with one skip and one unrelated failure:
`LocalizationParityTests.testEveryBundledLiteralHasACatalogKey` reports missing
keys in the existing model-storage discovery and migration screens. The new
Plan label has all 21 translations. The first export-snapshot attempt also
hit a WebKit snapshot lifecycle error; the ordinary export test passes in
the full suite and the final focused run.

Workspace build passed. Workspace tests, formatting, and Clippy reached
concurrent Q3_K changes outside this UI scope: a high-bit assertion in
`crates/compute/tests/quant_gguf.rs`, formatting in that test and `q3_k.rs`,
and an f32/f16 type mismatch in `dequant_q3_k_gemv_parity.rs`. Those files
were not modified by this UI work. No real-model gate is required because
this change does not touch inference or numeric behavior.
