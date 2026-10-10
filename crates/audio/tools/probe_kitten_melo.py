"""Scratch probe: build tiny kitten_tts and melotts models, print weight
shapes and intermediate shapes through a forward pass.

Run from ../mlx-audio:
    ../.venv-mlxaudio/bin/python ../turbospark/crates/audio/tools/probe_kitten_melo.py
"""

import sys

import mlx.core as mx
import mlx.utils
import numpy as np

sys.path.insert(0, ".")

from mlx_audio.tts.models.kitten_tts.kitten_tts import Model as KittenModel, ModelConfig
from mlx_audio.tts.models.kitten_tts.modules import AlbertModelArgs

from mlx_audio.tts.models.melotts.melotts import Model as MeloModel, ModelConfig as MeloConfig


def seed_tree(module):
    tree = {}
    for k, v in mlx.utils.tree_flatten(module.parameters()):
        if v.dtype in (mx.int32, mx.int64, mx.uint32):
            tree[k] = v
        else:
            n = float(np.abs(np.asarray(v)).sum()) + 1.0
            tree[k] = (mx.random.normal(v.shape) * 0.05 + n * 0.001).astype(mx.float32)
    return tree


# ---------------- kitten ----------------
config = ModelConfig(
    hidden_dim=32,
    max_conv_dim=32,
    max_dur=8,
    n_layer=2,
    n_mels=80,
    n_token=120,
    style_dim=128,
    text_encoder_kernel_size=5,
    asr_res_dim=4,
    plbert=dict(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=24,
        intermediate_size=32,
        max_position_embeddings=128,
        embedding_size=8,
        inner_group_num=1,
        num_hidden_groups=1,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
    ),
    istftnet=dict(
        resblock_kernel_sizes=[3, 5, 7],
        upsample_rates=[4, 4],
        upsample_initial_channel=32,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes=[8, 8],
        gen_istft_n_fft=32,
        gen_istft_hop_size=4,
    ),
    sample_rate=24000,
)
kmodel = KittenModel(config)
kmodel.eval()
mx.random.seed(0)
np.random.seed(0)
kmodel.load_weights(list(seed_tree(kmodel).items()), strict=True)

print("== kitten params ==")
for k, v in mlx.utils.tree_flatten(kmodel.parameters()):
    if any(s in k for s in ["ups.0", "ups.1", "F0_conv", "asr_res", "conv_post", "noise_convs"]):
        print(k, v.shape)

ids = mx.array([[0, 5, 9, 3, 42, 7, 0]], dtype=mx.int32)
ref_s = mx.random.normal((1, 2 * config.style_dim))
audio = kmodel(ids, ref_s, speed=1.0)
print("kitten audio", audio.shape)

# Instrument decoder shapes
dec = kmodel.decoder
print("ups:", [(u.weight_v.shape, u.stride, u.padding) for u in dec.generator.ups])
print("noise_convs:", [(c.weight.shape, c.stride, c.padding) for c in dec.generator.noise_convs])

# Trace generator stages manually
asr = mx.random.normal((1, config.hidden_dim, 9))
f0 = mx.random.normal((1, 18))
n = mx.random.normal((1, 18))
s = ref_s[:, : config.style_dim]
f0c = dec.F0_conv(f0[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
print("f0_conv out", f0c.shape)
x = mx.concatenate([asr, f0c, n], axis=1)
print("decoder x", x.shape)
xe = dec.encode(x, s)
print("encode out", xe.shape)
asr_res = dec.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
print("asr_res", asr_res.shape)
res = True
for i, block in enumerate(dec.decode):
    if res:
        x2 = mx.concatenate([xe, asr_res, f0c, n], axis=1)
    else:
        x2 = xe
    xe = block(x2, s)
    print("decode block", i, x2.shape, "->", xe.shape)
    if getattr(block, "upsample_type", "none") != "none":
        res = False

gen = dec.generator
f0u = gen.f0_upsamp(f0[:, None].transpose(0, 2, 1))
print("f0_upsamp", f0.shape, "->", f0u.shape)
har, noi, uv = gen.m_source(f0u)
print("har source", har.shape, "noise", noi.shape)
har_s = mx.squeeze(har.transpose(0, 2, 1), axis=1)
mag, ph = gen.stft.transform(har_s)
print("stft transform", mag.shape, ph.shape)
harc = mx.concatenate([mag, ph], axis=1).swapaxes(2, 1)
print("har concat", harc.shape)
xg = xe
for i in range(gen.num_upsamples):
    xg = gen.ups[i](xg.swapaxes(2, 1), mx.conv_transpose1d).swapaxes(2, 1)
    print("gen up", i, xg.shape)
xs = gen.noise_convs[i](harc.swapaxes(2, 1))
print("noise conv out", xs.shape)

# ---------------- melo ----------------
mconfig = MeloConfig(
    sampling_rate=44100,
    filter_length=64,
    hop_length=16,
    segment_size=8192,
    add_blank=True,
    n_speakers=4,
    spk2id={"EN-Default": 0, "EN-US": 1},
    inter_channels=32,
    hidden_channels=32,
    filter_channels=64,
    n_heads=2,
    n_layers=2,
    n_layers_trans_flow=2,
    kernel_size=3,
    p_dropout=0.1,
    resblock="1",
    resblock_kernel_sizes=[3, 5, 7],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[4, 4, 2],
    upsample_initial_channel=32,
    upsample_kernel_sizes=[8, 8, 4],
    n_layers_q=2,
    gin_channels=16,
    use_spk_conditioned_encoder=True,
    use_transformer_flow=True,
    num_tones=8,
    num_languages=4,
    n_vocab=64,
    bert_hidden_size=1024,
)
mmodel = MeloModel(mconfig)
mmodel.eval()
mx.random.seed(1)
np.random.seed(1)
mmodel.load_weights(list(seed_tree(mmodel).items()), strict=True)

print("== melo params ==")
for k, v in mlx.utils.tree_flatten(mmodel.parameters()):
    if any(s in k for s in ["ups.0", "ups.2", "conv_pre", "conv_post", "dec.cond", "enc_p.bert_proj", "emb_g"]):
        print(k, v.shape)

n_phones = 9
x = mx.array([[i % mconfig.n_vocab for i in range(n_phones)]], dtype=mx.int32)
x_lengths = mx.array([n_phones])
sid = mx.array([0])
tone = mx.array([[7] * n_phones], dtype=mx.int32)
language = mx.array([[2] * n_phones], dtype=mx.int32)
bert = mx.zeros((1, 1024, n_phones))
ja = mx.zeros((1, 768, n_phones))
audio = mmodel.infer(x, x_lengths, sid, tone, language, bert, ja_bert=ja, noise_scale=0.1)
print("melo audio", audio.shape)
