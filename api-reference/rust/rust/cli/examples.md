---
description: "Runnable turbospark-cli command lines grouped by job: text generation, chat, models, images, music, voice cloning, and the server."
icon: play
---

# Examples

These command lines use flags that the parser accepts. Replace the model names, paths, and repo IDs with your own. Models must be installed before generation, so start with [Install and remove models](#install-and-remove-models) if you have none.

The examples were not run as part of writing this page. Output is described only where the source fixes its shape. See [Exit status and output](exit-status-and-output.md) before you script any of them.

## Generate text from one prompt

```sh
turbospark-check --model gemma4 --prompt "Explain quantum decoherence in one sentence."
```

Use `--prompt` for base or raw-completion models. It skips the chat template. An instruction-tuned model answers poorly in this mode, so use a conversation file for those models (see below).

Stdout begins with a `resolved invocation:` block. It lists `model`, `mode`, `max_new`, `max_context`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `seed`, `stop`, `rdadvise`, `expert_cache_slots`, `prefill_chunk`, `power_profile`, `max_tokens_per_sec`, `reasoning`, `kv_bits`, and `quiet`, followed by a `note:` line. Generated text comes after a `generating (real forward pass):` line.

For a deterministic run, add `--temperature 0` (greedy decoding) and a seed. Also test with a sampled run. Greedy output does not check the sampler.

```sh
turbospark-check --model gemma4 --prompt "Name three rivers." --temperature 0
turbospark-check --model gemma4 --prompt "Name three rivers." --temperature 0.7 --top-p 0.9 --seed 42
```

## Generate from a conversation file

Instruction-tuned models need the chat template, which `--messages-file` applies. Put the conversation in a JSON array:

```json
[
  {"role": "system", "content": "Answer in one sentence."},
  {"role": "user", "content": "Why is the sky blue?"}
]
```

```sh
turbospark-check --model gemma4 --messages-file ./messages.json --max-new 256 --seed 7
```

Roles are `system`, `developer`, `user`, `assistant`, and `tool`. Any other role is an error.

For a vision-capable model, `content` can be a list of parts. The image path is read from the `path` field:

```json
[
  {"role": "user", "content": [
    {"type": "image", "path": "page.png"},
    {"type": "text", "text": "Transcribe this."}
  ]}
]
```

```sh
turbospark-check --model VISION_MODEL --messages-file ./page.json
```

Use a model that has a vision tower. `--image PATH` attaches images from the command line to a `--messages-file` conversation that has no image parts. `--image-batch` runs one prompt per image. Using `--image` together with image parts in the file is refused. `--image` with `--prompt` is also refused at run time.

## Chat in the terminal

```sh
turbospark-check --model tinyllama --chat --system "Be brief."
```

`turbospark run` is shorter. With no prompt and no mode flag, it starts the chat:

```sh
turbospark run gemma4
turbospark run gemma4 "Explain quantum computing"     # one-shot prompt, same as --prompt
turbospark run gemma4 --temperature 0.7 --max-new 256
```

Inside chat, type `/clear` to reset history, `/history` to print it on stderr, and `/quit` to leave. Ctrl-D also leaves. `--image` is accepted with `--chat` but is not sent to the model in this source, so use `--messages-file` for images.

## Rewrite a draft

```sh
turbospark-check --model humanizer-12b --humanize-file draft.txt --max-new 512 --seed 3
turbospark humanize humanizer-12b draft.txt
cat draft.txt | turbospark humanize humanizer-12b -
```

The rewrite goes to stdout and the audit notes go to stderr. Stdout also starts with the resolved block and a `humanizing (real forward pass):` header, so remove those lines before you use the text. The model's sampling settings are fixed in this mode, so the sampling flags do not change the result.

## Steer with a control vector

Steering applies a `.gguf` control vector. Each `--steering` path is one vector, and the other steering flags pair with paths by position:

```sh
turbospark-check --model gemma4 --chat \
  --steering ./refusal.gguf --steering-mode ablate --steering-scale 1.0 \
  --steering-layers 12:20
```

A steering flag without `--steering` is refused. So is a steering value with no matching path. The option table has the full rules in [Commands overview](commands-overview.md#steering).

## Install and remove models

```sh
turbospark-model list
turbospark-model list --filter gemma
turbospark-model info gemma4
turbospark-model path gemma4
turbospark-model pull gemma4
turbospark-model pull gemma4 --out /Volumes/models/gemma4
turbospark-model rm gemma4            # asks you to type the alias
turbospark-model rm gemma4 --yes      # no prompt
```

`list` prints a header row (`ALIAS STATUS KIND DOWNLOAD MODEL`) and marks installed rows with `*`. `path` prints one directory, which makes it safe to use in `$(...)`.

A pull streams the whole checkpoint and cannot resume. If it fails, start it again.

Speech and music profiles have their own commands:

```sh
turbospark-model list-speech
turbospark-model list-audio
turbospark-model pull-audio minimax-music3-4bit
turbospark-model auth --status
```

Save a Hugging Face token with `auth --set`. It reads the token from stdin, so pipe it in rather than typing it on the command line:

```sh
printf '%s\n' "$HF_TOKEN" | turbospark-model auth --set
```

## Check a repo before installing it

`probe` reads only the repo headers. It does not download weights. It exits `1` when the engine would refuse the model, so you can gate a pull on it:

```sh
turbospark-model probe Qwen/Qwen3-30B-A3B-GGUF && \
  turbospark-model pull --repo Qwen/Qwen3-30B-A3B-GGUF --alias qwen3-30b
```

A pinned revision goes after `@`. Use `--force` only when you have decided to install past a refusal. GGUF repos often need `--sidecar-repo` for tokenizer files, and `--file` picks one file from a repo that has several.

## Pick a model for this machine

```sh
turbospark-model recommend
turbospark-model recommend --context 8192
turbospark-model recommend --context 8192 --load-guard balanced
turbospark-model recommend --budget 36G
turbospark-model recommend --probe
```

`recommend` ranks rows against this machine's memory. Rank under the same `--load-guard` tier that the session will open with. A ranking under `relaxed` can promise a fit that `strict` will refuse. `--budget` overrides the memory probe, and a bare number is bytes.

## Generate an image

```sh
turbospark-image generate --model z-image-turbo \
  --prompt "A cinematic photo of an astronaut on Mars during sunset" \
  --output mars.png --seed 42

turbospark-image generate --model z-image-turbo --prompt "A red bicycle" \
  --output bike.png --width 768 --height 512
```

Install the image model first with `turbospark-model pull-image --alias z-image-turbo`. The output path must not exist. On success the command prints the path, so you can capture it with `OUT=$(...)`.

Pack a Diffusers export into an install directory:

```sh
turbospark-image pack --source ./diffusers-export --output ./zimage-install \
  --model-id ORG/MODEL --model-revision REVISION_40_HEX
```

The `turbospark image --prompt "..."` form from the crate README omits the required `--model` and `--output`. Use the `generate` form above.

## Generate music

```sh
turbospark-model pull-audio minimax-music3-4bit

turbospark-music generate --model minimax-music3-4bit \
  --caption "fast upbeat synthwave with punchy drums" \
  --lyrics "[instrumental]" --output song.wav --seed 1

turbospark-music generate --model minimax-music3-4bit \
  --caption "warm acoustic folk" --lyrics-file lyrics.txt \
  --output folk.wav --timings
```

The output file is overwritten without a prompt. `--timings` writes per-stage timing to stderr. `--wav-format float32` writes samples without rounding.

## Clone a voice

Use only a voice you have the right to use. The `--consent` flag is required, and the run is refused without it.

```sh
turbospark-clone speak --model ./megatts3-converted \
  --reference voice.wav --text "Hello world, this is a test." \
  --output out.wav --consent --quality fast --seed 1

turbospark-clone speak --model ./megatts3-converted \
  --reference voice.wav --text-file script.txt \
  --output out.wav --consent --steps 24 --cpu
```

`--quality` and `--steps` cannot be combined. `--text-file` is capped at 32768 bytes. The text must be English.

## Run the server and connect an agent

The server is `turbospark-server`, which this crate does not build. The front end finds it next to `turbospark` or on `PATH`.

```sh
turbospark start --port 9000         # background daemon
turbospark status                    # Status: RUNNING, with PID, port, endpoint, log file
turbospark stop

turbospark start claude --model gemma4 --dry-run   # print the launch plan only
turbospark start claude --model gemma4             # start or reuse the server, then launch
turbospark gemma4 claude                           # same launch, model first
```

Arguments after `--` go to the agent unchanged. This keeps an agent's own flags from being read by the front end:

```sh
turbospark start claude --model gemma4 -- AGENT_ARGS
```

`start` without an agent name passes its arguments to `turbospark-server`. `restart` stops the running server and calls `start` again with the same arguments:

```sh
turbospark start --model gemma4 --port 9000
turbospark restart --model gemma4 --port 9000
```

## Version and help

```sh
turbospark --version          # prints "turbospark" and the version
turbospark-check --version
turbospark-check --help       # exit 0, usage on stdout
turbospark-image --help       # exit 0, usage on stdout
turbospark-music --help       # exit 2, usage on stderr (see the quirks list)
```

## Limits

- Generation needs macOS with a Metal device. On other platforms, `turbospark-check` prints the resolved request and stops.
- `serve`, `start`, `restart`, and `bench` depend on binaries this crate does not build. The flags they accept are documented with those binaries.
- Use `--temperature 0` for deterministic runs, and check sampled output too. Greedy output does not exercise the sampler.
