"""Dump pinned MLX Kokoro tensors and seeded PCM from an existing checkpoint.

PYTHONPATH=<mlx-audio e1b19b9054bf163f5d812221a54fcc346f1890e9>
reference-env/bin/python this-file <checkpoint-dir> <output-dir>
Uses MLX 0.31.2. Never downloads assets or modifies the installed checkpoint.
"""
import hashlib
import json
import sys
import subprocess
import inspect
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_audio.tts.models.kokoro.kokoro import Model, ModelConfig
from mlx_audio.tts.models.kokoro import istftnet
from mlx_audio.tts.models.kokoro.modules import LSTM
assert mx.__version__ == '0.31.2', 'Kokoro fixtures require MLX 0.31.2'
reference_root=Path(inspect.getsourcefile(LSTM)).resolve().parents[4]
reference_revision=subprocess.check_output(['git','-C',str(reference_root),'rev-parse','HEAD'],text=True).strip()
assert reference_revision == 'e1b19b9054bf163f5d812221a54fcc346f1890e9', 'wrong reference clone revision'


checkpoint, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
config = json.loads((checkpoint/'config.json').read_text())
model = Model(ModelConfig.from_dict(config))
weights = mx.load(str(checkpoint/'kokoro-v1_0.safetensors'))
assert all(v.dtype == mx.float32 for v in weights.values()), 'checkpoint dtype changed'
model.load_weights(list(model.sanitize(weights).items()), strict=True)
model.eval()
mx.eval(model.parameters())
voice = mx.load(str(checkpoint/'voices/af_heart.safetensors'))['voice']
traces = {}

def save(name, tensor):
    mx.eval(tensor)
    traces[name] = np.asarray(tensor, dtype=np.float32).reshape(-1).tolist()

class Capture:
    def __init__(self, inner, name, tuple_result=False):
        self.inner, self.name, self.tuple_result = inner, name, tuple_result
    def __call__(self, *args, **kwargs):
        result = self.inner(*args, **kwargs)
        save(self.name, result[0] if self.tuple_result else result)
        return result

# Capture actual recurrent inputs/outputs and independently computed input projections.
lstm_names={id(module):name for name,module in model.named_modules() if isinstance(module,LSTM)}
original_lstm = LSTM.__call__
def capture_lstm(self,x,*args,**kwargs):
    name=lstm_names[id(self)]
    save(name+'.lstm_in',x)
    for suffix in ['forward','backward']:
        bias=getattr(self,'bias_ih_'+suffix)+getattr(self,'bias_hh_'+suffix)
        save(name+'.projection.'+suffix,mx.addmm(bias,x,getattr(self,'Wx_'+suffix).T))
    result=original_lstm(self,x,*args,**kwargs)
    save(name+'.lstm_out',result[0])
    return result
LSTM.__call__=capture_lstm

# Instrument the actual pinned SineGen scan and interpolation operators.
original_interpolate = istftnet.interpolate
source_interpolations = 0
def source_interpolate(*args, **kwargs):
    global source_interpolations
    result = original_interpolate(*args, **kwargs)
    if args[0].shape[1] == 9:
        name = 'source_rad_small' if source_interpolations == 0 else 'source_phase_up'
        save(name, result.transpose(0,2,1)[:,:39600,:])
        source_interpolations += 1
    return result
istftnet.interpolate = source_interpolate
original_cumsum = mx.cumsum
def capture_scan(x, *args, **kwargs):
    result = original_cumsum(x, *args, **kwargs)
    if x.ndim == 3 and x.shape[-1] == 9:
        save('source_cumsum', result)
    return result
mx.cumsum = capture_scan

model.bert = Capture(model.bert, 'bert', True)
model.bert_encoder = Capture(model.bert_encoder, 'duration_features')
model.text_encoder = Capture(model.text_encoder, 'text_encoder')
original = model.predictor.F0Ntrain
def prosody(*args, **kwargs):
    result = original(*args, **kwargs)
    save('f0', result[0]); save('noise', result[1])
    return result
model.predictor.F0Ntrain = prosody
model.decoder.encode = Capture(model.decoder.encode, 'decoder_encode')
for i, block in enumerate(model.decoder.decode):
    model.decoder.decode[i] = Capture(block, 'decoder_block'+str(i))
model.decoder.generator.m_source = Capture(model.decoder.generator.m_source, 'harmonic_source', True)
original_transform = model.decoder.generator.stft.transform
def capture_transform(*args, **kwargs):
    magnitude, phase = original_transform(*args, **kwargs)
    save('harmonic_spectrum', mx.concatenate([magnitude, phase],axis=1))
    return magnitude, phase
model.decoder.generator.stft.transform = capture_transform
for i, block in enumerate(model.decoder.generator.noise_convs):
    class ConvCapture:
        def __init__(self, inner, name): self.inner, self.name = inner, name
        def __call__(self, *args, **kwargs):
            result = self.inner(*args, **kwargs)
            save(self.name,result.transpose(0,2,1))
            return result
    model.decoder.generator.noise_convs[i] = ConvCapture(block,'noise_conv'+str(i))
for i, block in enumerate(model.decoder.generator.noise_res):
    model.decoder.generator.noise_res[i] = Capture(block, 'generator_noise'+str(i))
# ConvWeighted returns channels-last; the portable model traces channel-major.
original_post = model.decoder.generator.conv_post
class PostCapture:
    def __call__(self, *args, **kwargs):
        result = original_post(*args, **kwargs)
        save('conv_post', result.transpose(0,2,1))
        return result
model.decoder.generator.conv_post = PostCapture()
text = 'Hello, world!'
phonemes = 'h\u0259l\u02c8O, w\u02c8\u025c\u0279ld!'
mx.random.seed(0)
result = model(phonemes, voice[len(phonemes)-1], speed=1.0, return_output=True)
save('durations', result.pred_dur)
save('pcm', result.audio)
record = dict(style=np.asarray(voice[len(phonemes)-1],dtype=np.float32).reshape(-1)[:128].tolist(), reference_revision=reference_revision, mlx_version=mx.__version__, checkpoint_revision='a71e4d38b236d968966a2002c4c895dbd12b1c3c', text=text, phonemes=phonemes, style_row=len(phonemes)-1, seed=0, speed=1.0, traces=traces)
for name in ['config.json','kokoro-v1_0.safetensors','voices/af_heart.safetensors']:
    record.setdefault('asset_sha256',{})[name] = hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
(output/'reference.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='traces'},indent=2))
print({name:len(values) for name,values in traces.items()})
