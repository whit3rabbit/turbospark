# Audio API (CLI server)

`turbospark serve` can serve speech-to-text, text-to-speech, and music
generation next to chat. It is the standalone server only: the Swift app's
embedded server registers no audio routes. Machine-readable contracts are
available at [openapi/turbospark.openapi.yaml](openapi/turbospark.openapi.yaml)
(full server) and [openapi/audio.openapi.yaml](openapi/audio.openapi.yaml)
(audio routes), and tests keep them in step with the router.

## Start it

```sh
turbospark serve \
  --stt-model whisper-base-en \
  --tts-model ~/models/kokoro-82m \
  --music-model minimax-music3-4bit \
  --api-key "$KEY"
```

Each flag takes an install directory or an installed alias, and may repeat.
With none of `--model`, `--embedding-model`, or `--model-dir`, the server is
audio-only. Audio routes exist only when at least one audio flag is given.
Otherwise they answer 404. Models can also be attached and detached while
running (see Models).

There is no TLS. Set `--api-key`, and note that `--bind tailnet` requires one.

## Speech to text

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "authorization: Bearer $KEY" \
  -F file=@clip.wav -F model=whisper-base-en -F response_format=verbose_json
```

- Input is WAV only: PCM at 8, 16, 24, or 32 bits, or float32, at any rate
  and channel count. It is downmixed and resampled to 16 kHz mono. The limit
  is 128 MiB and 30 minutes. Anything else is 415.
- `response_format`: `json` (default), `verbose_json`, `text`, `srt`, `vtt`.
- A raw `audio/wav` body works too: `POST ...?model=ID` with `--data-binary`.
- Options the runners cannot honour are refused with 400: a non-empty
  `prompt`, a non-zero `temperature`, word timestamps, unknown fields.
- `language` is a tag such as `en`, or `auto`. Whisper returns segments per
  30 s window. Qwen3-ASR returns one clip-level segment and echoes the
  requested language.
- `stream=true` returns NDJSON: `start`, one `segment` line per segment, then
  `complete` (or one `error`). The runners decode a whole clip at a time, so
  segment lines arrive together when decoding ends. `start` is sent at once
  so a proxy sees bytes during a long decode.
- `async=true` returns 202 and a job (see Jobs).

### Realtime (WebSocket)

`GET /v1/audio/transcriptions/realtime?model=ID` upgrades to a WebSocket.
Send binary frames of 16 kHz mono PCM (`s16le`, or `f32le` with
`&encoding=f32le`), then `{"type":"commit"}`. The reply is
`{"type":"complete","segment_id":0,"text":"...","segments":[...]}`.

This is utterance-level. The client decides where an utterance ends, there is
no server-side endpointing, and there is no partial text. Closing the socket
discards uncommitted audio. Limits: 300 s buffered per socket, 1 MiB per
frame, 60 s idle, 8 sockets.

Authentication is the ordinary header on the upgrade request, so browsers
cannot connect. A key in the query string is deliberately not accepted.

```sh
websocat -H="Authorization: Bearer $KEY" \
  "ws://127.0.0.1:8080/v1/audio/transcriptions/realtime?model=whisper-base-en"
```

## Text to speech

```sh
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model":"kokoro-82m","input":"Hello there.","response_format":"wav"}' \
  -o hello.wav
```

Kokoro, US English, voice `af_heart` only. `input` (alias `text`) is at most
4096 characters, `speed` 0.5 to 2.0. Output is 24 kHz mono: `wav` (default) or
`pcm` (16-bit little-endian). OpenAI's default is mp3. This server cannot
encode it and refuses by name. `stream=true` needs `pcm` and sends one chunk
per synthesized segment. A failure after the first chunk ends the body early.

## Music generation

```sh
curl http://127.0.0.1:8080/v1/audio/generate \
  -H "authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model":"minimax-music3-4bit","prompt":"slow lofi piano","audio_length":20,"async":true}'
```

MiniMax Music 3 needs macOS and Metal. A full-size request takes minutes or
longer, so use `async`. `lyrics` defaults to `[instrumental]`. `audio_length`
is 0 to 360 s and `num_inference_steps` is 1 to 30. Fields the model has no
control for (`audio_start`, `guidance_scale`, ...) are refused.

Output is 44.1 kHz stereo 16-bit WAV. The repository records only short smoke
runs of this model, so treat output quality as unqualified.

## Jobs

`async=true` on a transcription or generation returns 202 with `job_id`.
`GET /v1/audio/jobs/ID` reports `running`, `succeeded`, `failed`, or
`cancelled` (`running` includes queued). `GET .../result` returns what the
synchronous call would have: 409 while running, 410 if cancelled. `DELETE`
cancels or deletes.

Jobs live in memory. At most 8 are unfinished, results are kept for 1 hour or
256 MiB, and everything is lost on restart. Cancelling drops work still
waiting for a model. A job a model already started runs to the end and its
result is discarded.

## Models and metrics

- `GET /v1/audio/models` lists attached audio models. They also appear in
  `GET /v1/models` with `capabilities` of `speech_to_text`,
  `text_to_speech` or `music_generation`.
- `POST /v1/audio/models {"model_id":"ALIAS"}` attaches an installed
  catalog alias. Paths are refused because the route is reachable over HTTP.
  `DELETE /v1/audio/models/ID` unloads (409 while jobs run).
- `GET /v1/metrics` is Prometheus text for the audio routes only:
  request counts and time by endpoint and status, in-flight, 429s, models,
  jobs by status.
- `GET /health` reports `ready` when any model, audio included, is attached.

## Limits worth knowing

- One worker thread per loaded model, with a short queue (STT and TTS 4,
  music 2). A full queue is 429 with `retry-after`. Requests to one model run
  one at a time.
- Audio work is not gated against chat generation on the same GPU. Running
  both at once shares the device unsynchronized, and no measurement of that
  exists here.
- No mp3, flac, ogg, or opus decoding or encoding. No diarization, word
  timestamps, or confidences. No translation: `/v1/audio/translations` is
  501.
- Served-HTTP behavior of real models is smoke-tested, not benchmarked. See
  [AUDIO.md](AUDIO.md) and `crates/audio/MODELS.md` for what each model has
  actually been verified to do.
