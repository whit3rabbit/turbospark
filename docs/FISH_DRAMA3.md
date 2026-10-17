# Fish Audio Drama 3

Fish Audio Drama 3 is a multi-speaker text-to-speech model with zero-shot
voice cloning. TurboSpark runs the pinned 4-bit MLX conversion on Metal and
streams its routed experts from disk, so a 22 GB checkpoint speaks from about
6 GiB of memory. Output is mono 44.1 kHz. Built with Fish Audio.

This page is the user guide. The facts, measurements, and refusals behind it
are in [`FISH_DRAMA3_PHASE0.md`](FISH_DRAMA3_PHASE0.md).

## License

Drama 3 is under the Fish Audio Community License 1.0:

- Free for research and non-commercial use.
- Commercial use only while your annual revenue is under USD 500,000.
- No hosted service (an API or app whose main purpose is text to speech for
  third parties) without a separate license from Fish Audio.
- Every copy you distribute carries `LICENSE` and `NOTICE`.
- A product that uses the model shows "Built with Fish Audio".

The install keeps `LICENSE` and `NOTICE` beside the model. Installing asks you
to accept these terms, and cloning a voice asks you to confirm you have the
right to use it.

## Install

```sh
turbospark-model pull-drama3 --out ~/models/fish-drama3-4bit --accept-fish-license
```

The install converts the checkpoint on your Mac. It needs about 25 GiB free,
checks that before it downloads anything, and takes tens of minutes.

The five small files (config, tokenizer, codec, `LICENSE`, `NOTICE`) download
whole against pinned hashes. The six shards stream through the converter by
range read and are never stored whole. The command refuses an existing
destination and builds in a sibling `.partial` directory that it renames into
place. Ctrl-C stops the walk and removes the partial directory.

The catalog profile `fish-drama3-4bit` installs through the same walk with a
receipt in the managed store (`ts_audio_install_with_options`, with the
`fish_audio_community_license` acknowledgment). The receipt records the size
and sha256 of every converted file. The macOS app does not list it yet.

## Takes that stop early

Some takes speak the first sentence, a turn, or the words before a
`[pause]`, and then only silence until the frame cap
(`docs/FISH_DRAMA3_PHASE0.md`, sampling recipe). The default sampler is Fish
Audio's serving recipe (temperature 1.0, top-p 0.9, top-k 30), which traps
far fewer takes than mlx-audio's 0.7 and 0.7: 4 of 12 seeds of
`Welcome back. This is a short voiceover test with no reference voice.`
went silent under mlx-audio's recipe, against 0 of 12. The runtime reports a
take that still stopped early as `stopped_early` in the audio job result,
and a host can retake it with another seed.

## Speak

```sh
turbospark-speak --model ~/models/fish-drama3-4bit \
  --text "[sighs] Fine. If we miss it, it's on you." --output fine.wav
```

Delivery cues go inline, such as `[whispers]` or `[laughs]`. Speaker tags
`<|speaker:0|>` through `<|speaker:15|>` start a turn. There is no speed
control and no instruction field.

To clone a voice, pass a recording of 1 to 30 seconds, the exact words spoken
in it, and your confirmation:

```sh
turbospark-speak --model ~/models/fish-drama3-4bit \
  --reference me.wav --reference-text "What I said in the recording." --consent \
  --text "Something new, in that voice." --output cloned.wav
```

`--seed` repeats a run. `--temperature`, `--top-p`, `--top-k` and
`--max-frames` override the reference sampler recipe (0.7, 0.7, 30, 1024).
`--max-frames` only shortens a chunk. A text-scaled cap still applies beneath
it, because some texts run away without one.

## HTTP

```sh
turbospark-server --tts-model ~/models/fish-drama3-4bit
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"fish-drama3-4bit","input":"Hello there."}' -o hello.wav
```

The model id is the install directory's name. Output is 44.1 kHz mono, `wav`
or `pcm`, and the rate is in the `x-audio-sample-rate` header.

With a `reference_audio` the request is a clone and needs `consent: true` and
`reference_text`. `temperature`, `top_p`, `top_k` and `max_frames` are
accepted. `steps`, `speed` other than 1.0, `instruction`, `p_w` and `t_w` are
refused by name. The full field rules are in [`AUDIO_API.md`](AUDIO_API.md).

## Native sessions and C

A session opened on a Drama 3 install takes the same request fields. The
reference audio is the job's PCM. The wire rules are in the "Fish Audio Drama
3" and "Derived installs" paragraphs of `crates/ffi/include/turbospark.h`.

## Limits and costs

- Metal only. There is no portable fallback.
- Peak memory on an M4 Max at 16 streamed expert slots was 6,106 to 6,296 MiB
  over a session that spoke 44 s of text, then cloned a voice. Admission
  budgets 8.2 GiB. The protocol and the three readings are in the Drama 3 row
  of [`BENCHMARKS.md`](BENCHMARKS.md).
- Warm, generation runs at about 1.15 times real time, with the first samples
  after about 2 s. Streamed playback keeps up once it starts.
- Opening a managed install hashes every file before it loads, about 24 GB for
  Drama 3.
- Only the 4-bit conversion is supported. The 3-bit and bf16 conversions are
  refused until each has its own gates.
- One reference voice per request.
