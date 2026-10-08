"""Independent MLX 0.31.2 fixtures. Set PYTHONPATH to the pinned mlx-audio clone.

Reference: e1b19b9054bf163f5d812221a54fcc346f1890e9. Regeneration only,
never called by Rust tests. Run reference-env/bin/python this-file output.json.
"""
import json
import sys
import subprocess
import inspect
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_audio.tts.models.kokoro.modules import LSTM
assert mx.__version__ == '0.31.2', 'Kokoro fixtures require MLX 0.31.2'
reference_root=Path(inspect.getsourcefile(LSTM)).resolve().parents[4]
reference_revision=subprocess.check_output(['git','-C',str(reference_root),'rev-parse','HEAD'],text=True).strip()
assert reference_revision == 'e1b19b9054bf163f5d812221a54fcc346f1890e9', 'wrong reference clone revision'



def values(shape, offset=0):
    return mx.array((np.sin(np.arange(np.prod(shape)) * 0.37 + offset) * 0.2).astype(np.float32).reshape(shape))


def flat(x):
    mx.eval(x)
    return np.asarray(x).reshape(-1).tolist()


rows = []
for seq, ic, oc in [(3, 7, 5), (35, 64, 33), (1, 128, 64)]:
    x, w, b = values((seq, ic)), values((oc, ic), 1), values((oc,), 2)
    rows.append(dict(op='linear', seq=seq, ic=ic, oc=oc, x=flat(x), w=flat(w), bias=flat(b), expected=flat(mx.addmm(b, x, w.T))))
for seq, ic, oc, kernel, stride, padding, dilation, groups, transpose in [
    (7, 4, 6, 3, 2, 2, 2, 2, False),
    (5, 4, 4, 3, 2, 0, 1, 4, True),
    (9, 3, 5, 4, 2, 1, 1, 1, True),
    (37, 4, 33, 3, 1, 1, 1, 1, False),
]:
    x = values((1, seq, ic))
    shape = (ic, oc // groups, kernel) if transpose else (oc, ic // groups, kernel)
    w, b = values(shape, 1), values((oc,), 2)
    if transpose:
        mw = w.transpose(0, 2, 1) if groups == ic == oc else w.transpose(1, 2, 0)
        y = mx.conv_transpose1d(x, mw, stride=stride, padding=padding, dilation=dilation, groups=groups) + b
    else:
        y = mx.conv1d(x, w.transpose(0, 2, 1), stride=stride, padding=padding, dilation=dilation, groups=groups) + b
    rows.append(dict(op='conv', seq=seq, ic=ic, oc=oc, kernel=kernel, stride=stride, padding=padding, dilation=dilation, groups=groups, transpose=transpose, x=flat(x.transpose(0,2,1)), w=flat(w), bias=flat(b), expected=flat(y.transpose(0,2,1))))
for backward in [False, True]:
    seq, ic, hidden = 5, 4, 3
    model = LSTM(ic, hidden)
    x, wx, wh, bi, bh = values((1,seq,ic)), values((4*hidden,ic),1), values((4*hidden,hidden),2), values((4*hidden,),3), values((4*hidden,),4)
    suffix = 'backward' if backward else 'forward'
    setattr(model, 'Wx_'+suffix, wx)
    setattr(model, 'Wh_'+suffix, wh)
    setattr(model, 'bias_ih_'+suffix, bi)
    setattr(model, 'bias_hh_'+suffix, bh)
    y, _ = (model._backward_direction if backward else model._forward_direction)(x)
    projection = mx.addmm(bi+bh, x, wx.T)
    rows.append(dict(op='lstm', seq=seq, ic=ic, hidden=hidden, backward=backward, x=flat(x), wx=flat(wx), wh=flat(wh), bias=flat(bi+bh), projection=flat(projection), expected=flat(y)))
rng = []
for seed in [0,7,2**32+3]:
    mx.random.seed(seed)
    phase = mx.random.uniform(shape=(1,9))
    noise = mx.random.normal((1,17,9))
    unused = mx.random.normal((1,17,1))
    next_phase = mx.random.uniform(shape=(1,9))
    mx.random.seed(seed)
    mx.random.uniform(shape=(1,9))
    normal_uniform = mx.random.uniform(low=np.nextafter(np.float32(-1),np.float32(0)).item(),high=1.0,shape=(1,17,9))
    rng.append(dict(seed=seed, phase=flat(phase), noise=flat(noise), unused=flat(unused), next_phase=flat(next_phase), normal_uniform=flat(normal_uniform)))
scans=[]
for length,width in [(132,9),(35,2)]:
    x=(mx.abs(values((1,width,length),0.23))+0.25).transpose(0,2,1)
    scans.append(dict(rows=length,columns=width,x=flat(x),expected=flat(mx.cumsum(x,axis=1))))
from mlx_audio.dsp import hanning,stft
signal=values((300,),0.713)
window=hanning(20,periodic=True)
z=stft(signal,n_fft=20,hop_length=5,window=window)
spectrum=mx.concatenate([mx.abs(z).T,mx.arctan2(mx.imag(z),mx.real(z)).T])
stft_fixture=dict(signal=flat(signal),window=flat(window),expected=flat(spectrum))
Path(sys.argv[1]).write_text(json.dumps(dict(scans=scans,stft=stft_fixture,reference_revision=reference_revision, mlx_version=mx.__version__, operators=rows, rng=rng), indent=2)+'\n')

# Portable audio tests package their own small draw fixture.
if len(sys.argv)>2:
    Path(sys.argv[2]).write_text(json.dumps(dict(reference_revision=reference_revision,mlx_version=mx.__version__,rng=rng),indent=2)+'\n')
