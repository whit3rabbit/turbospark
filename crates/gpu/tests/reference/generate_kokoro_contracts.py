"""Independent F32 execution fixtures, MLX 0.31.2 and pinned mlx-audio.

PYTHONPATH=<pinned sibling> reference-env/bin/python this-file output.json
Synthetic inputs are deterministic; expected tensors call MLX and upstream LSTM.
MLX is Apple MIT; mlx-audio is Blaizzy MIT. See Kokoro shader MIT notice.
Regeneration only. Rust tests do not invoke Python or download assets.
"""
import inspect
import json
from pathlib import Path
import subprocess
import sys
import mlx.core as mx
import numpy as np
from mlx_audio.tts.models.kokoro.modules import LSTM
assert mx.__version__ == '0.31.2'
root = Path(inspect.getsourcefile(LSTM)).resolve().parents[4]
revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == 'e1b19b9054bf163f5d812221a54fcc346f1890e9'

def value(shape, seed):
    return mx.array(np.random.default_rng(seed).normal(0, 0.2, shape).astype(np.float32))

def flat(x):
    mx.eval(x)
    return np.asarray(x).reshape(-1).tolist()

linear=[]
for i,(rows,ic,oc) in enumerate([(17,128,64),(17,135,64),(33,768,64),(1,128,1024),(71,9,1),(513,513,513)]):
    x,w,b=value((rows,ic),i),value((oc,ic),i+10),value((oc,),i+20)
    linear.append(dict(rows=rows,ic=ic,oc=oc,x=flat(x),w=flat(w),bias=flat(b),expected=flat(mx.addmm(b,x,w.T))))
norm=[]
for i,(rows,cols,columns) in enumerate([(3,512,False),(35,1031,False),(64,132,True),(64,269,True),(64,1031,True),(4,1031,True),(4,7,True),(3,31,False)]):
    x=value((1,cols,rows) if columns else (rows,cols),i+30)
    if columns: x=x.transpose(0,2,1)
    mean=mx.mean(x,axis=-1,keepdims=True)
    # Upstream mx.var materializes the squared differences, then row-reduces.
    variance=mx.var(x,axis=-1,keepdims=True)
    out=(x-mean)/mx.sqrt(variance+1e-5)
    norm.append(dict(rows=rows,cols=cols,columns=columns,x=flat(x),expected=flat(out)))
layer=value((3,768),40);w=value((768,),41);b=value((768,),42)
layernorm=dict(rows=3,cols=768,x=flat(layer),w=flat(w),bias=flat(b),expected=flat(mx.fast.layer_norm(layer,w,b,1e-12)))
conv=[]
for i,(seq,ic,oc,kernel,stride,padding,groups,transpose) in enumerate([(35,32,16,3,1,1,1,False),(13,4,4,3,2,0,4,True)]):
    x=value((1,seq,ic),50+i)
    shape=(ic,oc//groups,kernel) if transpose else (oc,ic//groups,kernel)
    w,b=value(shape,60+i),value((oc,),70+i)
    mw=w.transpose(0,2,1)
    out=(mx.conv_transpose1d if transpose else mx.conv1d)(x,mw,stride=stride,padding=padding,groups=groups)+b
    conv.append(dict(seq=seq,ic=ic,oc=oc,kernel=kernel,stride=stride,padding=padding,groups=groups,transpose=transpose,x=flat(x.transpose(0,2,1)),w=flat(w),bias=flat(b),expected=flat(out.transpose(0,2,1))))
lstm=[]
for i,backward in enumerate([False,True]):
    seq,ic,hidden=9,5,128
    model=LSTM(ic,hidden)
    x,wx,wh,bi,bh=value((1,seq,ic),80),value((4*hidden,ic),81),value((4*hidden,hidden),82),value((4*hidden,),83),value((4*hidden,),84)
    suffix='backward' if backward else 'forward'
    for name,v in [('Wx_',wx),('Wh_',wh),('bias_ih_',bi),('bias_hh_',bh)]:setattr(model,name+suffix,v)
    out,_=(model._backward_direction if backward else model._forward_direction)(x)
    lstm.append(dict(hidden=hidden,backward=backward,w=flat(wh),projection=flat(mx.addmm(bi+bh,x,wx.T)),expected=flat(out)))
u=value((1031,),90)
u=mx.clip(u*2,-0.999,0.999)
normal=mx.sqrt(mx.array(2.0))*mx.erfinv(u)
signal=value((1031,),91)*30
record=dict(reference_revision=revision,mlx_version=mx.__version__,linear=linear,norm=norm,layernorm=layernorm,conv=conv,lstm=lstm,unary=dict(x=flat(signal),sin=flat(mx.sin(signal)),tanh=flat(mx.tanh(signal))),normal=dict(x=flat(u),expected=flat(normal)))
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
