"""Matched-input precision fixtures from the pinned Kokoro checkpoint.

PYTHONPATH=<mlx-audio e1b19b9054bf163f5d812221a54fcc346f1890e9>
MLX 0.31.2 python this-file <existing checkpoint> <output.json>
No downloads. These small real inputs expose waveform-causal arithmetic changes.
"""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.tts.models.kokoro.kokoro import Model, ModelConfig
from mlx_audio.tts.models.kokoro.istftnet import weight_norm
assert mx.__version__ == '0.31.2'
root = Path(inspect.getsourcefile(Model)).resolve().parents[4]
revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == 'e1b19b9054bf163f5d812221a54fcc346f1890e9'
checkpoint, output = map(Path, sys.argv[1:])
config = json.loads((checkpoint/'config.json').read_text())
model = Model(ModelConfig.from_dict(config))
weights = mx.load(str(checkpoint/'kokoro-v1_0.safetensors'))
assert all(w.dtype == mx.float32 for w in weights.values())
model.load_weights(list(model.sanitize(weights).items()), strict=True)
model.eval()
mx.eval(model.parameters())
def flat(x):
    mx.eval(x)
    return np.asarray(x, dtype=np.float32).reshape(-1).tolist()
phonemes = 'h\u0259l\u02c8O, w\u02c8\u025c\u0279ld!'
ids = mx.array([[0] + [config['vocab'][p] for p in phonemes] + [0]])
e = model.bert.embeddings
raw = e.word_embeddings(ids) + e.position_embeddings(mx.arange(ids.shape[1])[None,:]) + e.token_type_embeddings(mx.zeros_like(ids))
normalized = e.LayerNorm(raw)
x = model.bert.encoder.embedding_hidden_mapping_in(normalized)
layer = model.bert.encoder.albert_layer_groups[0].albert_layers[0]
q, k, v = [getattr(layer.attention, name)(x) for name in ['query', 'key', 'value']]
# A complete head retains all keys and the independent materialized graph.
qq, kk, vv = [layer.attention.transpose_for_scores(a)[:, 0] for a in [q,k,v]]
scores = mx.matmul(qq, kk.transpose(0,2,1)) / (layer.attention.attention_head_size ** 0.5)
probabilities = mx.softmax(scores, axis=-1)
attention = mx.matmul(probabilities, vv)
ffn = layer.ffn(layer.attention(x))
# The first real row includes both signs and large values in the erf branch.
ffn = ffn[:, :1, :]
folds = []
for name in ['predictor.F0.0.conv1', 'predictor.F0.1.conv2', 'predictor.F0.1.pool', 'decoder.generator.ups.0']:
    w = weights[name+'.weight_v'][:4]
    g = weights[name+'.weight_g'][:4]
    expected = weight_norm(w.transpose(0,2,1), g, dim=0).transpose(0,2,1)
    folds.append(dict(name=name, shape=list(w.shape), x=flat(w), g=flat(g), expected=flat(expected)))
record = dict(reference_revision=revision, mlx_version=mx.__version__,
    asset_sha256={name:hashlib.sha256((checkpoint/name).read_bytes()).hexdigest() for name in ['config.json','kokoro-v1_0.safetensors']},
    norm=dict(rows=ids.shape[1], cols=raw.shape[-1], x=flat(raw), weight=flat(e.LayerNorm.weight), bias=flat(e.LayerNorm.bias), expected=flat(normalized)),
    attention=dict(seq=qq.shape[1],dim=qq.shape[2],q=flat(qq),k=flat(kk),v=flat(vv),scores=flat(scores),probabilities=flat(probabilities),expected=flat(attention)),
    gelu=dict(x=flat(ffn),expected=flat(nn.gelu(ffn))), folds=folds)
output.write_text(json.dumps(record, indent=2)+'\n')
print('pinned matched-input fixtures:', output, output.stat().st_size)
