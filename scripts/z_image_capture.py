#!/usr/bin/env python3
"""Staged pinned Diffusers captures. Each invocation owns one component.

Use the locked environment in z_image_reference_requirements.txt. Outputs
belong in target/ig0/runs/<case>; only JSON manifests go into evidence docs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
from transformers import AutoModel, AutoTokenizer

from z_image_evidence import Evidence, Monitor, preflight, sha256, validate_capture
from z_image_probe import validate_inventory

PROMPTS = {
    "empty": "",
    "composition": "A red ceramic teapot to the left of a blue cup on a wooden table, a window behind them.",
    "typography": 'A shop sign with the exact words "FRESH BREAD" in clear black letters.',
    "detail": "Macro photograph of a honeybee on lavender, fine hairs and translucent wing veins.",
    "lighting": "A lighthouse in winter at dusk, warm windows reflected on wet snow, cold blue shadows.",
    "unicode": "\u96ea\u4e2d\u306e\u706f\u53f0, caf\u00e9 au cr\u00e9puscule",
    "overlong": "small red lighthouse " * 600,
}


def pipeline(model, **kwargs):
    return ZImagePipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler"),
                          vae=None, text_encoder=None, tokenizer=None, transformer=None, **kwargs)


def verify_weights(model, inventory, component):
    report = json.loads(Path(inventory).read_text())
    validate_inventory(report)
    checked = {}
    for path, row in report["weights"].items():
        if path.startswith(component + "/"):
            actual = sha256(model / path)
            if actual != row["sha256"]:
                raise ValueError(f"weight checksum mismatch: {path}")
            checked[path] = actual
    for path, row in report["files"].items():
        if path.endswith(".json") or path.startswith("tokenizer/"):
            if sha256(model / path) != row["sha256"]:
                raise ValueError(f"config/tokenizer checksum mismatch: {path}")
    return checked


def encode(args, evidence):
    tok = AutoTokenizer.from_pretrained(args.model / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_pretrained(args.model / "text_encoder", dtype=torch.bfloat16,
                                        local_files_only=True).eval()
    if args.quantize:
        from z_image_quantization import apply_quantization
        evidence.data["quantization"] = apply_quantization(encoder, args.model, "text_encoder")
    encoder.to(args.device)
    # The base model avoids materializing unused vocabulary logits.
    encoder.config.use_cache = False
    pipe = pipeline(args.model)
    pipe.tokenizer, pipe.text_encoder = tok, encoder
    prompt = PROMPTS[args.case]
    framed = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
    tokens = tok(framed, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    evidence.save("token_ids", tokens.input_ids)
    evidence.save("attention_mask", tokens.attention_mask)
    evidence.data["framed_prompt"] = framed
    result = pipe.encode_prompt([prompt], device=args.device, do_classifier_free_guidance=False)[0][0]
    evidence.save("conditioning", result)


def denoise(args, evidence):
    evidence.input(args.out / "encode.json")
    embeds = torch.from_numpy(np.load(args.out / "conditioning.npy")).to(args.device, torch.bfloat16)
    model = ZImageTransformer2DModel.from_pretrained(args.model, subfolder="transformer",
                                                    torch_dtype=torch.bfloat16, local_files_only=True).eval()
    if args.quantize:
        from z_image_quantization import apply_quantization
        evidence.data["quantization"] = apply_quantization(model, args.model, "transformer")
    model.to(args.device)
    pipe = pipeline(args.model)
    pipe.transformer = model
    noise = torch.randn((1, 16, args.height // 8, args.width // 8),
                        generator=torch.Generator("cpu").manual_seed(args.seed), dtype=torch.float32)
    evidence.save("initial_noise", noise)
    calls = []
    def forward_hook(_, inputs, output):
        calls.append(1)
    model.register_forward_hook(forward_hook)
    # Capture only the first invocation per representative block to bound fixture size.
    for index in (0, len(model.layers) // 2, len(model.layers) - 1):
        name = f"block_{index:02}"
        def hook(_, inputs, output, name=name):
            if name + "_output" not in evidence.data["arrays"]:
                evidence.save(name + "_input", inputs[0])
                mask = inputs[1]
                evidence.data[name + "_mask_was_none"] = mask is None
                if mask is None:
                    mask = torch.ones(inputs[0].shape[:2], dtype=torch.bool)
                evidence.save(name + "_mask", mask)
                evidence.save(name + "_freqs", inputs[2])
                evidence.save(name + "_modulation", inputs[3])
                evidence.save(name + "_output", output)
        model.layers[index].register_forward_hook(hook)
    updates = []
    def callback(_, i, t, kwargs):
        evidence.save(f"latent_{i:02}", kwargs["latents"])
        updates.append(float(t))
        return kwargs
    result = pipe(prompt_embeds=[embeds], latents=noise.to(args.device), width=args.width, height=args.height,
                  num_inference_steps=args.steps, guidance_scale=0.0, output_type="latent",
                  callback_on_step_end=callback).images
    evidence.save("final_latents", result)
    evidence.save("timesteps", pipe.scheduler.timesteps)
    evidence.save("sigmas", pipe.scheduler.sigmas)
    evidence.data.update(actual_forwards=len(calls), scheduler_updates=len(updates))


def decode(args, evidence):
    evidence.input(args.out / "denoise.json")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32,
                                        local_files_only=True).eval().to(args.device)
    latents = torch.from_numpy(np.load(args.out / "final_latents.npy")).to(args.device)
    result = vae.decode(latents / vae.config.scaling_factor + vae.config.shift_factor, return_dict=False)[0]
    evidence.save("decoded_pixels", result)
    from diffusers.image_processor import VaeImageProcessor
    image = VaeImageProcessor().postprocess(result, output_type="pil")[0]
    image.save(args.out / "image.png")
    evidence.data["png_sha256"] = sha256(args.out / "image.png")


def contracts(args, evidence):
    from diffusers import ModelMixin, ConfigMixin
    class Counter(ModelMixin, ConfigMixin):
        def __init__(self):
            super().__init__()
            self.register_to_config(in_channels=16)
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.in_channels = 16
            self.calls = 0
        def forward(self, x, *unused, **kwargs):
            self.calls += 1
            return ([torch.zeros_like(v) for v in x],)
    pipe = pipeline(args.model)
    pipe.transformer = Counter()
    rows = []
    for steps in (1, 8, 9):
        pipe.transformer.calls = 0
        result = pipe(prompt_embeds=[torch.zeros(1, 2560)], height=16, width=16,
                      latents=torch.zeros(1, 16, 2, 2), guidance_scale=0, num_inference_steps=steps,
                      output_type="latent")
        rows.append({"requested_steps": steps, "actual_forwards": pipe.transformer.calls,
                     "timesteps": pipe.scheduler.timesteps.tolist(), "sigmas": pipe.scheduler.sigmas.tolist()})
    evidence.data["schedules"] = rows
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
    scheduler.set_timesteps(sigmas=np.linspace(1, 1 / 9, 9).tolist())
    state = torch.tensor([1.0, -1.0], dtype=torch.float32)
    updates = []
    for t in scheduler.timesteps:
        state = scheduler.step(torch.tensor([0.25, -0.5]), t, state, return_dict=False)[0]
        updates.append(state.clone())
    evidence.save("scheduler_updates", torch.stack(updates))
    tok = AutoTokenizer.from_pretrained(args.model / "tokenizer", local_files_only=True)
    token_cases = {}
    for name, prompt in PROMPTS.items():
        frame = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        t = tok(frame, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        evidence.save(name + "_ids", t.input_ids)
        evidence.save(name + "_mask", t.attention_mask)
        token_cases[name] = {"untruncated_tokens": len(tok(frame).input_ids),
                            "retained_tokens": int(t.attention_mask.sum())}
    evidence.data["token_cases"] = token_cases
    dims = []
    for h, w in ((16, 16), (32, 48), (1024, 1024), (1023, 1024), (0, 16), (-16, 16)):
        try:
            pipe(prompt_embeds=[torch.zeros(1, 2560)], height=h, width=w, guidance_scale=0,
                 num_inference_steps=1, output_type="latent")
            dims.append({"height": h, "width": w, "accepted": True})
        except (ValueError, RuntimeError) as exc:
            dims.append({"height": h, "width": w, "accepted": False, "error": str(exc)})
    evidence.data["dimension_probes"] = dims


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["contracts", "encode", "denoise", "decode", "validate"])
    p.add_argument("--model", type=Path, default=Path("target/ig0/model"))
    p.add_argument("--inventory", type=Path, default=Path("docs/verification/z-image-ig0-inputs.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--case", choices=PROMPTS, default="lighting")
    p.add_argument("--device", choices=["cpu", "mps"], default="mps")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture-only", action="store_true",
                   help="allow busy AC hardware for correctness; exclude timings from benchmarks")
    p.add_argument("--quantize", action="store_true", help="emulate group-64 INT4 linear weights for quality comparisons")
    args = p.parse_args()
    if args.stage == "validate":
        manifests = list(args.out.glob("*.json"))
        names = {path.name for path in manifests}
        if names not in ({"contracts.json"}, {"encode.json", "denoise.json", "decode.json"}):
            raise ValueError("validation requires a complete contract or three-stage capture")
        for path in manifests:
            validate_capture(path)
        return
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16 or args.steps <= 0:
        p.error("positive dimensions divisible by 16 and positive steps are required")
    settings = {"prompt": PROMPTS[args.case], "case": args.case, "width": args.width, "height": args.height,
                "steps": args.steps, "seed": args.seed, "device": args.device,
                "noise_generator": "torch CPU float32", "guidance": 0.0}
    settings["quantized_linears"] = args.quantize
    evidence = Evidence(args.out, args.stage, settings)
    if args.stage != "contracts":
        try:
            evidence.data["preflight"] = preflight(require_quiet=not args.capture_only)
            evidence.data["benchmark_eligible"] = not args.capture_only
        except RuntimeError as exc:
            (args.out / (args.stage + ".failed.json")).write_text(json.dumps({
                "complete": False, "stage": args.stage, "error": str(exc)}, indent=2) + "\n")
            raise
        component = {"encode": "text_encoder", "denoise": "transformer", "decode": "vae"}[args.stage]
        evidence.data["verified_weights"] = verify_weights(args.model, args.inventory, component)
    if args.stage == "contracts":
        # Small CPU structural probes do not establish a performance envelope.
        with torch.inference_mode():
            contracts(args, evidence)
    else:
        with Monitor(args.device) as monitor, torch.inference_mode():
            globals()[args.stage](args, evidence)
            if args.device == "mps":
                torch.mps.synchronize()
        evidence.data["resources"] = monitor.report()
        evidence.data["benchmark_eligible"] &= evidence.data["resources"]["quiet_ac_throughout"]
    evidence.finish()
    (args.out / (args.stage + ".failed.json")).unlink(missing_ok=True)
    print(evidence.path)


if __name__ == "__main__":
    main()
