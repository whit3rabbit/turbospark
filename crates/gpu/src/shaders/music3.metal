#include <metal_stdlib>
using namespace metal;

// Portions adapted from MLX 0.32.3 erf and attention kernels.
// Copyright (c) 2023-2026 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// BF16 values use round-to-nearest-even, including the intermediate casts in MLX.
inline float music_round(float x, uint dtype) {
    if (dtype == 0) return x;
    if (dtype == 1) return float(half(x));
    uint u=as_type<uint>(x);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xffff0000u);
}

// Encoding: F32, F16, BF16, affine, MXFP4, MXFP8, NVFP4.
inline float music_e4m3(uint c) {
    uint e = (c >> 3) & 15, m = c & 7;
    float x = e == 0 ? float(m) * 0.001953125f : (1.0f + float(m)/8.0f) * exp2(float(int(e)-7));
    return c & 128 ? -x : x;
}
inline float music_unscaled(uint mode,uint code) {
    if(mode==5) return music_e4m3(code);
    const float table[8]={0,0.5f,1,1.5f,2,3,4,6};
    return code&8 ? -table[code&7] : table[code&7];
}
// Affine scales and offsets are uploaded at checkpoint width (bf16) when every
// value is exactly representable, and widened here: the conversion is a shift,
// so the value is bit-identical to the f32 upload it replaces.
#ifdef MUSIC3_SCALES_BF16
inline float music_sc(device const float *a, ulong i) {
    return as_type<float>(uint(((device const ushort*)a)[i])<<16);
}
#else
inline float music_sc(device const float *a, ulong i) { return a[i]; }
#endif
inline float music_value(device const uchar *w, device const float *sc,
                         device const float *off, device const uchar *bs,
                         constant uint *p, uint row, uint col) {
    uint mode=p[0], cols=p[1], bits=p[2], group=p[3];
    ulong index=ulong(row)*cols+col;
    if (mode==0) return ((device const float*)w)[index];
    if (mode==1) return float(((device const half*)w)[index]);
    if (mode==2) return as_type<float>(uint(((device const ushort*)w)[index])<<16);
    uint words=(cols*bits+31)/32, bit=col*bits, shift=bit%32;
    device const uint *u=(device const uint*)w+ulong(row)*words;
    uint code=u[bit/32]>>shift;
    if (shift+bits>32) code|=u[bit/32+1]<<(32-shift);
    code&=(1u<<bits)-1;
    ulong g=ulong(row)*(cols/group)+col/group;
    if (mode==3) return float(code)*music_sc(sc,g)+music_sc(off,g);
    float scale=mode==6 ? music_e4m3(bs[g]) : exp2(float(int(bs[g])-127));
    if (mode==5) return music_e4m3(code)*scale;
    const float table[8]={0,0.5f,1,1.5f,2,3,4,6};
    float x=table[code&7];
    return (code&8 ? -x : x)*scale;
}

kernel void music3_linear(device const uchar *w [[buffer(0)]],
 device const float *sc [[buffer(1)]], device const float *off [[buffer(2)]],
 device const uchar *bs [[buffer(3)]], device const float *x [[buffer(4)]],
 device const float *bias [[buffer(5)]], device float *y [[buffer(6)]],
 constant uint *p [[buffer(7)]], uint tid [[thread_index_in_threadgroup]],
 uint gid [[threadgroup_position_in_grid]]) {
    uint out=p[4], row=gid/out, o=gid%out, cols=p[1];
#ifdef KOKORO_CONTRACTS
    bool native_vector = p[6]!=0 || p[12];
#else
    bool native_vector = p[6]!=0;
#endif
    if(p[0]<3 && native_vector && p[7]==1 && cols>64 && cols<16*out) {
        // Native GEMV assigns four adjacent products to each lane before
        // the shuffle-down reduction. Strided products cross rounding ties.
        float result=0;
        if(tid<32) for(uint c=tid*4;c<cols;c+=128) {
            for(uint i=0;i<4 && c+i<cols;i++) result+=music_value(w,sc,off,bs,p,o,c+i)*x[c+i];
        }
        result+=simd_shuffle_down(result,ushort(16));result+=simd_shuffle_down(result,ushort(8));
        result+=simd_shuffle_down(result,ushort(4));result+=simd_shuffle_down(result,ushort(2));result+=simd_shuffle_down(result,ushort(1));
        if(tid==0) y[gid]=music_round(music_round(result,p[6])+(p[5] ? bias[o] : 0),p[6]);
        return;
    }
    if(p[0]==3 && p[6]!=0 && !p[9] && p[10]) {
        float result=0;
        if(tid<8) for(uint g=tid;g<cols/p[3];g+=8) {
            for(uint c=g*p[3];c<(g+1)*p[3];c+=8) {
                float acc=0;
                for(uint i=0;i<min(8u,(g+1)*p[3]-c);i++) acc=fma(x[ulong(row)*cols+c+i],music_value(w,sc,off,bs,p,o,c+i),acc);
                result+=acc;
            }
        }
        result+=simd_shuffle_down(result,ushort(4));result+=simd_shuffle_down(result,ushort(2));result+=simd_shuffle_down(result,ushort(1));
        if(tid==0) y[gid]=music_round(music_round(result,p[6])+(p[5] ? bias[o] : 0),p[6]);
        return;
    }
    if(p[0]>=4 && p[6]!=0 && !p[9] && !p[10] && p[11]) {
        // fp_qmv[_fast] assigns consecutive packed values to each SIMD lane.
        float result=0;
        if(tid<32) for(uint c=tid*p[11];c<cols;c+=32*p[11]) {
            float acc=0;
            for(uint i=0;i<p[11] && c+i<cols;i++) {
                uint bit=(c+i)*p[2],shift=bit%32;device const uint *u=(device const uint*)w+ulong(o)*((cols*p[2]+31)/32);
                uint code=(u[bit/32]>>shift)&((1u<<p[2])-1);
                acc+=x[ulong(row)*cols+c+i]*music_unscaled(p[0],code);
            }
            uint group=ulong(o)*(cols/p[3])+c/p[3];
            float scale=p[0]==6 ? music_e4m3(bs[group]) : exp2(float(int(bs[group])-127));
            result+=scale*acc;
        }
        result=simd_sum(result);
        if(tid==0) y[gid]=music_round(music_round(result,p[6])+(p[5] ? bias[o] : 0),p[6]);
        return;
    }
    if(p[0]>=4 && p[6]!=0 && !p[9] && p[10]) {
        // fp_qmv_wide keeps each group unscaled until its float4 dot sum.
        float result=0;
        if(tid<16) for(uint g=tid;g<cols/p[3];g+=16) {
            float scale=p[0]==6 ? music_e4m3(bs[ulong(o)*(cols/p[3])+g]) : exp2(float(int(bs[ulong(o)*(cols/p[3])+g])-127));
            float acc=0;
            for(uint c=g*p[3];c<(g+1)*p[3];c+=4) {
                float4 weights,inputs;
                for(uint i=0;i<4;i++) {
                    uint bit=(c+i)*p[2],shift=bit%32;device const uint *u=(device const uint*)w+ulong(o)*((cols*p[2]+31)/32);
                    uint code=(u[bit/32]>>shift)&((1u<<p[2])-1);
                    weights[i]=music_unscaled(p[0],code);
                    inputs[i]=x[ulong(row)*cols+c+i];
                }
                acc+=dot(weights,inputs);
            }
            result+=scale*acc;
        }
        result+=simd_shuffle_down(result,ushort(8));result+=simd_shuffle_down(result,ushort(4));
        result+=simd_shuffle_down(result,ushort(2));result+=simd_shuffle_down(result,ushort(1));
        if(tid==0) y[gid]=music_round(music_round(result,p[6])+(p[5] ? bias[o] : 0),p[6]);
        return;
    }
    threadgroup float parts[4];
    float result=0;
    for(uint part=0;part<p[8];part++) {
        uint begin=part*(cols/p[8]),end=begin+cols/p[8];
        float sum=0;
        if(p[0]==3 && p[6]!=0 && !p[9] && p[2]!=8 && !p[10]) {
            // qmv's four-element input sums use typed addition before bias*sum.
            for(uint g=begin/p[3]+tid;g<end/p[3];g+=128) {
                float dot=0,xsum=0;
                for(uint c=g*p[3];c<(g+1)*p[3];c+=4) {
                    float xs=0;
                    for(uint i=0;i<min(4u,(g+1)*p[3]-c);i++) {
                        float xv=x[ulong(row)*cols+c+i];
                        xs=music_round(xs+xv,p[6]);
                        uint bit=(c+i)*p[2],shift=bit%32;device const uint *u=(device const uint*)w+ulong(o)*((cols*p[2]+31)/32);
                        uint code=u[bit/32]>>shift; if(shift+p[2]>32) code|=u[bit/32+1]<<(32-shift);code&=(1u<<p[2])-1;
                        dot+=xv*float(code);
                    }
                    xsum+=xs;
                }
                // Explicit fma: left to the compiler, the pair of products fuses on an
                // M4 Max but not on every GPU, and the last bit moves.
                sum+=fma(dot,music_sc(sc,ulong(o)*(cols/p[3])+g),xsum*music_sc(off,ulong(o)*(cols/p[3])+g));
            }
        } else for(uint c=begin+tid;c<end;c+=128) {
            float weight=music_value(w,sc,off,bs,p,o,c);
            if(p[0]==3 && p[6]!=0 && p[9]) weight=music_round(weight,p[6]);
            sum=fma(x[ulong(row)*cols+c],weight,sum);
        }
        sum=simd_sum(sum);
        if(tid%32==0) parts[tid/32]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0) {
            float partial=parts[0]+parts[1]+parts[2]+parts[3];
            result=p[8]>1 ? music_round(result+music_round(partial,p[6]),p[6]) : result+partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0) {
        if(p[0]>=3 || p[7]==1) result=music_round(result,p[6]);
        y[gid]=music_round(result+(p[5] ? bias[o] : 0.0f),p[6]);
    }
}

// Packed form of the affine "wide" branch of music3_linear (qmv_fast
// emulation: eight lanes per output, lane t owns groups t, t+8, ..., each
// 8-element chunk is summed in order into `acc`, `acc` is added to the
// lane's running total, and lanes reduce with shuffle-down 4, 2, 1). The
// per-output arithmetic is unchanged; the gain is layout only: sixteen
// (row, output) pairs share one threadgroup instead of one pair using
// eight of 128 threads, and a four-bit chunk of eight codes is one 32-bit
// load with the group's scale and offset hoisted out of the element loop.
// Requires group % 8 == 0 and cols % group == 0 (checked on the host).
kernel void music3_linear_wide(device const uchar *w [[buffer(0)]],
 device const float *sc [[buffer(1)]], device const float *off [[buffer(2)]],
 device const uchar *bs [[buffer(3)]], device const float *x [[buffer(4)]],
 device const float *bias [[buffer(5)]], device float *y [[buffer(6)]],
 constant uint *p [[buffer(7)]], uint tid [[thread_index_in_threadgroup]],
 uint tg [[threadgroup_position_in_grid]]) {
    uint out=p[4], cols=p[1], bits=p[2], group=p[3];
    ulong pair=ulong(tg)*16+tid/8;
    uint lane=tid%8;
    bool live=pair<ulong(p[7])*out;
    float result=0;
    if(live) {
        uint row=uint(pair/out), o=uint(pair%out), groups=cols/group;
        device const uint *u=(device const uint*)w+ulong(o)*((cols*bits+31)/32);
        device const float *xr=x+ulong(row)*cols;
        for(uint g=lane;g<groups;g+=8) {
            float s=music_sc(sc,ulong(o)*groups+g), f=music_sc(off,ulong(o)*groups+g);
            for(uint c=g*group;c<(g+1)*group;c+=8) {
                float acc=0;
                if(bits==4) {
                    uint word=u[c/8];
                    float4 xa=*(device const float4*)(xr+c), xb=*(device const float4*)(xr+c+4);
                    float xv[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
                    for(uint i=0;i<8;i++) acc=fma(xv[i],fma(float((word>>(4*i))&15u),s,f),acc);
                } else {
                    for(uint i=0;i<8;i++) {
                        uint bit=(c+i)*bits,shift=bit%32;
                        uint code=u[bit/32]>>shift; if(shift+bits>32) code|=u[bit/32+1]<<(32-shift);
                        code&=(1u<<bits)-1;
                        acc=fma(xr[c+i],fma(float(code),s,f),acc);
                    }
                }
                result+=acc;
            }
        }
    }
    result+=simd_shuffle_down(result,ushort(4));result+=simd_shuffle_down(result,ushort(2));result+=simd_shuffle_down(result,ushort(1));
    if(live && lane==0) y[pair]=music_round(music_round(result,p[6])+(p[5] ? bias[uint(pair%out)] : 0),p[6]);
}

// Packed forms of the two float-packed (MXFP4, MXFP8, NVFP4) GEMV branches
// of music3_linear. Each (row, output) pair keeps its lanes, loop order,
// float4 `dot`, and reduction exactly; only the number of pairs per
// threadgroup changes. The branches ran one pair per 128-thread threadgroup
// with 16 (wide) or 32 (vector) lanes working.
//
// fp_qmv_wide: sixteen lanes per pair, eight pairs per threadgroup. Lane 0's
// shuffle-down tree (8, 4, 2, 1) reads only lanes 0..15 of its own pair.
kernel void music3_linear_fp_wide(device const uchar *w [[buffer(0)]],
 device const float *sc [[buffer(1)]], device const float *off [[buffer(2)]],
 device const uchar *bs [[buffer(3)]], device const float *x [[buffer(4)]],
 device const float *bias [[buffer(5)]], device float *y [[buffer(6)]],
 constant uint *p [[buffer(7)]], uint tid [[thread_index_in_threadgroup]],
 uint tg [[threadgroup_position_in_grid]]) {
    uint out=p[4], cols=p[1];
    // MXFP8: decode the 256 codes once with music_e4m3 itself (each value
    // is the one the per-element call produced) instead of an exp2 per
    // weight. Every thread reaches the barrier.
    threadgroup float e4m3[256];
    if(p[0]==5) {
        for(uint i=tid;i<256;i+=128) e4m3[i]=music_e4m3(i);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ulong pair=ulong(tg)*8+tid/16;
    uint lane=tid%16;
    bool live=pair<ulong(p[7])*out;
    float result=0;
    if(live) {
        uint row=uint(pair/out), o=uint(pair%out);
        for(uint g=lane;g<cols/p[3];g+=16) {
            float scale=p[0]==6 ? music_e4m3(bs[ulong(o)*(cols/p[3])+g]) : exp2(float(int(bs[ulong(o)*(cols/p[3])+g])-127));
            float acc=0;
            for(uint c=g*p[3];c<(g+1)*p[3];c+=4) {
                float4 weights,inputs;
                for(uint i=0;i<4;i++) {
                    uint bit=(c+i)*p[2],shift=bit%32;device const uint *u=(device const uint*)w+ulong(o)*((cols*p[2]+31)/32);
                    uint code=(u[bit/32]>>shift)&((1u<<p[2])-1);
                    weights[i]=p[0]==5 ? e4m3[code] : music_unscaled(p[0],code);
                    inputs[i]=x[ulong(row)*cols+c+i];
                }
                acc+=dot(weights,inputs);
            }
            result+=scale*acc;
        }
    }
    result+=simd_shuffle_down(result,ushort(8));result+=simd_shuffle_down(result,ushort(4));
    result+=simd_shuffle_down(result,ushort(2));result+=simd_shuffle_down(result,ushort(1));
    if(live && lane==0) y[pair]=music_round(music_round(result,p[6])+(p[5] ? bias[uint(pair%out)] : 0),p[6]);
}

// fp_qmv[_fast]: thirty-two lanes (one simdgroup) per pair, four pairs per
// threadgroup, so simd_sum reduces exactly the pair's lanes.
kernel void music3_linear_fp_vector(device const uchar *w [[buffer(0)]],
 device const float *sc [[buffer(1)]], device const float *off [[buffer(2)]],
 device const uchar *bs [[buffer(3)]], device const float *x [[buffer(4)]],
 device const float *bias [[buffer(5)]], device float *y [[buffer(6)]],
 constant uint *p [[buffer(7)]], uint tid [[thread_index_in_threadgroup]],
 uint tg [[threadgroup_position_in_grid]]) {
    uint out=p[4], cols=p[1];
    // MXFP8 codes decoded once per threadgroup, as in music3_linear_fp_wide.
    threadgroup float e4m3[256];
    if(p[0]==5) {
        for(uint i=tid;i<256;i+=128) e4m3[i]=music_e4m3(i);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ulong pair=ulong(tg)*4+tid/32;
    uint lane=tid%32;
    bool live=pair<ulong(p[7])*out;
    float result=0;
    if(live) {
        uint row=uint(pair/out), o=uint(pair%out);
        for(uint c=lane*p[11];c<cols;c+=32*p[11]) {
            float acc=0;
            for(uint i=0;i<p[11] && c+i<cols;i++) {
                uint bit=(c+i)*p[2],shift=bit%32;device const uint *u=(device const uint*)w+ulong(o)*((cols*p[2]+31)/32);
                uint code=(u[bit/32]>>shift)&((1u<<p[2])-1);
                acc+=x[ulong(row)*cols+c+i]*(p[0]==5 ? e4m3[code] : music_unscaled(p[0],code));
            }
            uint group=ulong(o)*(cols/p[3])+c/p[3];
            float scale=p[0]==6 ? music_e4m3(bs[group]) : exp2(float(int(bs[group])-127));
            result+=scale*acc;
        }
    }
    result=simd_sum(result);
    if(live && lane==0) y[pair]=music_round(music_round(result,p[6])+(p[5] ? bias[uint(pair%out)] : 0),p[6]);
}

// MLX 0.32's `gemv_wide` (mlx/backend/metal/kernels/gemv.h): out[M, N] = x[M, K] @ w[N, K]^T
// for 2 to 15 vectors against a dense bf16 or f16 matrix on M3 and later. Each of
// k_lanes lanes owns every k_lanes-th float4 chunk of K, adds eight chunk dots at a
// time into a block sum, folds the block sums in order, then reduces the lanes with
// halving shuffles. The rows of a simdgroup stay separate when k_lanes is 16.
// The result for one vector does not depend on how vectors are grouped, so only
// k_lanes (p[14]) matters. Rounds once to the activation dtype.
inline float4 music_w4(device const ushort *row,uint idx,uint mode) {
    float4 r;
    for(uint i=0;i<4;i++) {
        ushort u=row[idx*4+i];
        r[i]=mode==2 ? as_type<float>(uint(u)<<16) : float(as_type<half>(u));
    }
    return r;
}
kernel void music3_gemv_wide(device const uchar *w [[buffer(0)]],
 device const float *sc [[buffer(1)]],device const float *off [[buffer(2)]],
 device const uchar *bs [[buffer(3)]],device const float *x [[buffer(4)]],
 device const float *bias [[buffer(5)]],device float *y [[buffer(6)]],
 constant uint *p [[buffer(7)]],uint gid [[threadgroup_position_in_grid]],
 uint simd_gid [[simdgroup_index_in_threadgroup]],uint simd_lid [[thread_index_in_simdgroup]]) {
    uint mode=p[0],K=p[1],N=p[4],M=p[7],k_lanes=p[14];
    uint rps=32/k_lanes;
    uint k_lane=simd_lid%k_lanes,sg_row=simd_lid/k_lanes;
    uint out_row=gid*(4*rps)+simd_gid*rps+sg_row;
    uint row=min(out_row,N-1);
    device const ushort *wr=(device const ushort*)w+ulong(row)*K;
    uint n_v4=K/4,n_main=n_v4-n_v4%(k_lanes*8);
    float result[16];
    for(uint v=0;v<16;v++) result[v]=0;
    for(uint base=0;base<n_main;base+=k_lanes*8) {
        float4 wf[8];
        for(uint i=0;i<8;i++) wf[i]=music_w4(wr,base+i*k_lanes+k_lane,mode);
        for(uint v=0;v<M;v++) {
            float acc=0;
            device const float *xv=x+ulong(v)*K;
            for(uint i=0;i<8;i++) {
                uint c=(base+i*k_lanes+k_lane)*4;
                acc+=dot(wf[i],float4(xv[c],xv[c+1],xv[c+2],xv[c+3]));
            }
            result[v]+=acc;
        }
    }
    for(uint idx=n_main+k_lane;idx<n_v4;idx+=k_lanes) {
        float4 wf=music_w4(wr,idx,mode);
        for(uint v=0;v<M;v++) {
            device const float *xv=x+ulong(v)*K;
            uint c=idx*4;
            result[v]+=dot(wf,float4(xv[c],xv[c+1],xv[c+2],xv[c+3]));
        }
    }
    for(uint v=0;v<M;v++) {
        for(ushort o=k_lanes/2;o>=1;o>>=1) result[v]+=simd_shuffle_down(result[v],o);
    }
    if(k_lane==0 && out_row<N) {
        for(uint v=0;v<M;v++) y[ulong(v)*N+out_row]=music_round(result[v],p[6]);
    }
}

kernel void music3_embedding(device const uchar *w [[buffer(0)]],
 device const float *sc [[buffer(1)]], device const float *off [[buffer(2)]],
 device const uchar *bs [[buffer(3)]], device const int *ids [[buffer(4)]],
 device float *y [[buffer(5)]], constant uint *p [[buffer(6)]],
 uint gid [[thread_position_in_grid]]) {
    y[gid]=music_value(w,sc,off,bs,p,uint(ids[gid/p[1]]),gid%p[1]);
}

// Native convolution unfolds kernel positions before channels into MMA tiles.
kernel void music3_conv_mma(device const uchar *w [[buffer(0)]],device const float *x [[buffer(1)]],device const float *bias [[buffer(2)]],device float *y [[buffer(3)]],constant uint *p [[buffer(4)]],uint lane [[thread_index_in_threadgroup]],uint gid [[threadgroup_position_in_grid]]) {
    uint ic=p[0],oc=p[1],ksize=p[2],stride=p[3],pad=p[4],dilation=p[5],transpose=p[6],len=p[7],outlen=p[8],blocks=(oc+7)/8,t0=(gid/blocks)*8,o0=(gid%blocks)*8;
    threadgroup float a[64],b[64],result[64];
    simdgroup_float8x8 am,bm,cm(0.0f);
    for(uint f0=0;f0<ic*ksize;f0+=8) {
        for(uint j=0;j<2;j++) {
            uint idx=lane+j*32,t=t0+idx/8,f=f0+idx%8,k=f/ic,i=f%ic;
            if(transpose) k=ksize-1-k;
            int pos=int(t*stride+k*dilation)-int(pad);
            bool valid=t<outlen && f<ic*ksize;
            if(transpose) {pos=int(t+pad)-int(k);valid=valid && pos>=0 && uint(pos)%stride==0;pos/=int(stride);}
            a[idx]=valid && pos>=0 && uint(pos)<len ? x[ulong(i)*len+uint(pos)] : 0;
            uint o=o0+idx%8,ff=f0+idx/8,kk=ff/ic,ii=ff%ic;
            if(transpose) kk=ksize-1-kk;
            ulong wi=transpose ? (ulong(ii)*oc+o)*ksize+kk : (ulong(o)*ic+ii)*ksize+kk;
            b[idx]=o<oc && ff<ic*ksize ? (p[11]==0 ? ((device const float*)w)[wi] : p[11]==1 ? float(((device const half*)w)[wi]) : as_type<float>(uint(((device const ushort*)w)[wi])<<16)) : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(am,a,8);simdgroup_load(bm,b,8);simdgroup_multiply_accumulate(cm,am,bm,cm);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(cm,result,8);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<2;j++) {uint idx=lane+j*32,t=t0+idx/8,o=o0+idx%8;if(t<outlen && o<oc) y[ulong(o)*outlen+t]=music_round(music_round(result[idx],p[10])+(p[9] ? bias[o] : 0),p[10]);}
}

// Tiled form of music3_conv_mma (see music3_linear_tiled). Each output keeps
// the same K-ordered chain of 8x8x8 products, including the zero-padded
// products for taps that miss the input; the gather index math and weight
// reads are shared across a (time x channel) tile instead of repeated per
// 8x8 tile. SRxSO must be 4.
template <uint SR, uint SO, uint RM, uint RN>
kernel void music3_conv_tiled(device const uchar *w [[buffer(0)]],device const float *x [[buffer(1)]],
 device const float *bias [[buffer(2)]],device float *y [[buffer(3)]],constant uint *p [[buffer(4)]],
 uint tid [[thread_index_in_threadgroup]],uint gid [[threadgroup_position_in_grid]]) {
    constexpr uint SGR=8*RM,SGO=8*RN,SGE=SGR*SGO,PER=SGE/32,TR=SR*SGR,TO=SO*SGO,BK=32;
    uint ic=p[0],oc=p[1],ksize=p[2],stride=p[3],pad=p[4],dilation=p[5],transpose=p[6],len=p[7],outlen=p[8];
    uint total_k=ic*ksize,oblocks=(oc+TO-1)/TO,t0=(gid/oblocks)*TR,o0=(gid%oblocks)*TO;
    uint simd=tid/32,lane=tid%32,sr=simd/SO,so=simd%SO;
    threadgroup float xs[TR*BK],ws[BK*TO],res[SR*SO*SGE];
    // p[14] K partitions (1 unless MLX runs this conv as an unfold plus a split-K
    // GEMM): each is its own float32 chain from zero, summed in order after.
    uint parts=p[14];
    float total[PER];
    for(uint m=0;m<PER;m++) total[m]=0;
    for(uint part=0;part<parts;part++) {
    simdgroup_float8x8 c[RM][RN];
    for(uint i=0;i<RM;i++) for(uint j=0;j<RN;j++) c[i][j]=simdgroup_float8x8(0.0f);
    uint begin=part*(total_k/parts),end=begin+total_k/parts;
    for(uint f0=begin;f0<end;f0+=BK) {
        for(uint idx=tid;idx<TR*BK;idx+=128) {
            uint r=idx%TR,kc=idx/TR,t=t0+r,f=f0+kc,k=f/ic,i=f%ic;
            // MLX's implicit-GEMM conv loader visits the taps inside each 16-channel
            // chunk (its channel jump rewinds the taps), so K runs chunk-major, not
            // tap-major. p[13] selects it: Kokoro always, Music 3 when the shape
            // takes MLX's implicit-GEMM path (see the host).
            if(p[13] && !transpose && ic%16==0) { k=(f/16)%ksize;i=(f/(16*ksize))*16+f%16; }
            // MLX's unfold walks the taps of a transposed conv in ascending weight
            // order (w[0] first), so no flip: the pairing w[k] with x[(t+pad-k)/stride]
            // is unchanged, only the summation order.
            int pos=int(t*stride+k*dilation)-int(pad);
            bool valid=t<outlen && f<end;
            if(transpose) {pos=int(t+pad)-int(k);valid=valid && pos>=0 && uint(pos)%stride==0;pos/=int(stride);}
            xs[r*BK+kc]=valid && pos>=0 && uint(pos)<len ? x[ulong(i)*len+uint(pos)] : 0;
        }
        for(uint idx=tid;idx<TO*BK;idx+=128) {
            uint oo=idx%TO,kc=idx/TO,o=o0+oo,ff=f0+kc,kk=ff/ic,ii=ff%ic;
            if(p[13] && !transpose && ic%16==0) { kk=(ff/16)%ksize;ii=(ff/(16*ksize))*16+ff%16; }
            ulong wi=transpose ? (ulong(ii)*oc+o)*ksize+kk : (ulong(o)*ic+ii)*ksize+kk;
            ws[kc*TO+oo]=o<oc && ff<end ? (p[11]==0 ? ((device const float*)w)[wi] : p[11]==1 ? float(((device const half*)w)[wi]) : as_type<float>(uint(((device const ushort*)w)[wi])<<16)) : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kb=0;kb<BK/8;kb++) {
            if(f0+kb*8>=end) break;
            simdgroup_float8x8 a[RM],b[RN];
            for(uint i=0;i<RM;i++) simdgroup_load(a[i],xs+(sr*SGR+i*8)*BK+kb*8,BK);
            for(uint j=0;j<RN;j++) simdgroup_load(b[j],ws+(kb*8)*TO+so*SGO+j*8,TO);
            for(uint i=0;i<RM;i++) for(uint j=0;j<RN;j++) simdgroup_multiply_accumulate(c[i][j],a[i],b[j],c[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<RM;i++) for(uint j=0;j<RN;j++) simdgroup_store(c[i][j],res+simd*SGE+i*8*SGO+j*8,SGO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint m=0;m<PER;m++) {
        float r=res[simd*SGE+(lane+32*m)%SGR*SGO+(lane+32*m)/SGR];
        total[m]=parts>1 ? total[m]+r : r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Lanes cover consecutive time steps so the channel-major store is contiguous.
    for(uint m=0;m<PER;m++) {
        uint e=lane+32*m,rr=e%SGR,oo=e/SGR,t=t0+sr*SGR+rr,o=o0+so*SGO+oo;
        if(t<outlen && o<oc) y[ulong(o)*outlen+t]=music_round(music_round(total[m],p[10])+(p[9] ? bias[o] : 0),p[10]);
    }
}
template [[host_name("music3_conv_tiled_32x32")]] [[kernel]] decltype(music3_conv_tiled<2,2,2,2>) music3_conv_tiled<2,2,2,2>;

// Polyphase form of music3_conv_tiled for a transposed convolution with
// ic % 8 == 0 and a single K partition. Output t = j*stride + phase only sees
// the taps k = r0 + stride*m (r0 = (phase+pad) % stride), at input j + c - m;
// every other tap is a block of exact zero products in music3_conv_tiled
// (ic % 8 == 0 keeps each 8-wide MMA block inside one tap), and adding an
// all-zero block leaves the accumulator unchanged. Dropping those blocks and
// keeping the taps in ascending order therefore gives the same bits for
// 1/stride of the work. A tile is 32 consecutive j of one phase.
template <uint SR, uint SO, uint RM, uint RN>
kernel void music3_convt_phase(device const uchar *w [[buffer(0)]],device const float *x [[buffer(1)]],
 device const float *bias [[buffer(2)]],device float *y [[buffer(3)]],constant uint *p [[buffer(4)]],
 uint tid [[thread_index_in_threadgroup]],uint gid [[threadgroup_position_in_grid]]) {
    constexpr uint SGR=8*RM,SGO=8*RN,SGE=SGR*SGO,PER=SGE/32,TR=SR*SGR,TO=SO*SGO,BK=32;
    uint ic=p[0],oc=p[1],ksize=p[2],stride=p[3],pad=p[4],len=p[7],outlen=p[8];
    uint oblocks=(oc+TO-1)/TO,jtiles=((outlen+stride-1)/stride+TR-1)/TR;
    uint o0=(gid%oblocks)*TO,j0=((gid/oblocks)%jtiles)*TR,phase=gid/(oblocks*jtiles);
    uint simd=tid/32,lane=tid%32,sr=simd/SO,so=simd%SO;
    uint r0=(phase+pad)%stride,c=(phase+pad-r0)/stride;
    uint taps=r0<ksize ? (ksize-r0+stride-1)/stride : 0,total_k=taps*ic;
    threadgroup float xs[TR*BK],ws[BK*TO],res[SR*SO*SGE];
    simdgroup_float8x8 acc[RM][RN];
    for(uint i=0;i<RM;i++) for(uint j=0;j<RN;j++) acc[i][j]=simdgroup_float8x8(0.0f);
    for(uint q0=0;q0<total_k;q0+=BK) {
        for(uint idx=tid;idx<TR*BK;idx+=128) {
            uint r=idx%TR,q=q0+idx/TR,j=j0+r,m=q/ic,i=q%ic;
            int pos=int(j)+int(c)-int(m);
            bool valid=q<total_k && j*stride+phase<outlen && pos>=0 && uint(pos)<len;
            xs[r*BK+idx/TR]=valid ? x[ulong(i)*len+uint(pos)] : 0;
        }
        for(uint idx=tid;idx<TO*BK;idx+=128) {
            uint oo=idx%TO,q=q0+idx/TO,o=o0+oo,m=q/ic,ii=q%ic,kk=r0+stride*m;
            ulong wi=(ulong(ii)*oc+o)*ksize+kk;
            ws[(idx/TO)*TO+oo]=o<oc && q<total_k ? (p[11]==0 ? ((device const float*)w)[wi] : p[11]==1 ? float(((device const half*)w)[wi]) : as_type<float>(uint(((device const ushort*)w)[wi])<<16)) : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kb=0;kb<BK/8;kb++) {
            if(q0+kb*8>=total_k) break;
            simdgroup_float8x8 a[RM],b[RN];
            for(uint i=0;i<RM;i++) simdgroup_load(a[i],xs+(sr*SGR+i*8)*BK+kb*8,BK);
            for(uint j=0;j<RN;j++) simdgroup_load(b[j],ws+(kb*8)*TO+so*SGO+j*8,TO);
            for(uint i=0;i<RM;i++) for(uint j=0;j<RN;j++) simdgroup_multiply_accumulate(acc[i][j],a[i],b[j],acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<RM;i++) for(uint j=0;j<RN;j++) simdgroup_store(acc[i][j],res+simd*SGE+i*8*SGO+j*8,SGO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint m=0;m<PER;m++) {
        uint e=lane+32*m,rr=e%SGR,oo=e/SGR,t=(j0+sr*SGR+rr)*stride+phase,o=o0+so*SGO+oo;
        float total=res[simd*SGE+rr*SGO+oo];
        if(t<outlen && o<oc) y[ulong(o)*outlen+t]=music_round(music_round(total,p[10])+(p[9] ? bias[o] : 0),p[10]);
    }
}
template [[host_name("music3_convt_phase_32x32")]] [[kernel]] decltype(music3_convt_phase<2,2,2,2>) music3_convt_phase<2,2,2,2>;

// Channel-major activations, PyTorch convolution weight layouts.
kernel void music3_conv(device const uchar *w [[buffer(0)]],
 device const float *x [[buffer(1)]], device const float *bias [[buffer(2)]],
 device float *y [[buffer(3)]], constant uint *p [[buffer(4)]],
 uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]]) {
    uint ic=p[0],oc=p[1],ksize=p[2],stride=p[3],pad=p[4],dilation=p[5],
         transpose=p[6],len=p[7],outlen=p[8],o=gid/outlen,t=gid%outlen;
    float sum=0;
    for(uint f=tid;f<ic*ksize;f+=128) {
        uint i=f/ksize,k=f%ksize;
        int pos=int(t*stride+k*dilation)-int(pad);
        ulong wi=(ulong(o)*ic+i)*ksize+k;
        if(transpose) {
            pos=int(t+pad)-int(k);
            if(pos<0 || uint(pos)%stride!=0) continue;
            pos/=int(stride);
            wi=(ulong(i)*oc+o)*ksize+k;
        }
        if(pos>=0 && uint(pos)<len) {
            float weight=p[11]==0 ? ((device const float*)w)[wi] : p[11]==1 ? float(((device const half*)w)[wi]) : as_type<float>(uint(((device const ushort*)w)[wi])<<16);
            sum+=x[ulong(i)*len+uint(pos)]*weight;
        }
    }
    sum=simd_sum(sum);
    threadgroup float parts[4];
    if(tid%32==0) parts[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
        float result=parts[0]+parts[1]+parts[2]+parts[3];
        if(p[10]!=0) result=music_round(result,p[10]);
        y[gid]=music_round(result+(p[9] ? bias[o] : 0.0f),p[10]);
    }
}

// Online softmax avoids materializing the quadratic attention matrix.
kernel void music3_attention(device const float *q [[buffer(0)]],
 device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
 device float *y [[buffer(3)]], constant uint *p [[buffer(4)]],
 device float *scratch [[buffer(5)]],
 uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]]) {
    uint batch=p[0],qlen=p[1],klen=p[2],heads=p[3],kh=p[4],dim=p[5],
         h=gid%heads,t=(gid/heads)%qlen,b=gid/(heads*qlen),kvh=h/(heads/kh);
    ulong qi=(ulong(b)*qlen*heads+t*heads+h)*dim;
    bool fallback=p[10]!=0 && !(dim==64 || dim==80 || dim==96 || dim==128 || dim==256);
    float scale=rsqrt(float(dim));
    float a=tid<dim ? q[qi+tid] : 0, acc=0,m=-INFINITY,denom=0;
    if (p[10]!=0) a=fallback ? music_round(a*music_round(scale,p[10]),p[10]) : a*scale;
    threadgroup float parts[8];
    uint groups=p[9]/32;
    uint count=p[7] ? min(klen,t+p[8]+1) : klen;
    for(uint s=0;s<count;s++) {
        ulong ki=p[6] ? (ulong(s)*batch*kh+b*kh+kvh)*dim : (ulong(b)*klen*kh+s*kh+kvh)*dim;
        float dot=simd_sum(tid<dim ? a*k[ki+tid] : 0.0f);
        if(tid%32==0) parts[tid/32]=dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score=0;
        for(uint g=0;g<groups;g++) score+=parts[g];
        if(p[10]==0) score*=scale;
        if(fallback) {
            if(tid==0) scratch[ulong(gid)*klen+s]=music_round(score,p[10]);
            threadgroup_barrier(mem_flags::mem_device);
            continue;
        }
        float next=max(m,score),old=exp(m-next),cur=exp(score-next);
        acc=acc*old+(tid<dim ? v[ki+tid]*cur : 0.0f);
        denom=denom*old+cur;m=next;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(fallback) {
        float maximum=-INFINITY;
        for(uint s=0;s<count;s++) maximum=max(maximum,scratch[ulong(gid)*klen+s]);
        float total=0;
        for(uint s=0;s<count;s++) total+=exp(scratch[ulong(gid)*klen+s]-maximum);
        for(uint s=0;s<count;s++) {
            ulong ki=p[6] ? (ulong(s)*batch*kh+b*kh+kvh)*dim : (ulong(b)*klen*kh+s*kh+kvh)*dim;
            float prob=music_round(exp(scratch[ulong(gid)*klen+s]-maximum)/total,p[10]);
            acc+=tid<dim ? prob*v[ki+tid] : 0.0f;
        }
        if(tid<dim) y[qi+tid]=music_round(acc,p[10]);
    } else if(tid<dim) y[qi+tid]=music_round(acc/denom,p[10]);
}

// Reduction and intermediate casts follow MLX 0.32.3 normalization.metal.
// MLX kernels are Copyright 2024 Apple Inc., MIT licensed.
inline float music_group_sum(float value, threadgroup float *parts, uint tid) {
    if(tid<32) parts[tid]=0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value=simd_sum(value);
    if(tid%32==0) parts[tid/32]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result=simd_sum(parts[tid%32]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}
kernel void music3_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
 device const float *bias [[buffer(2)]], device float *y [[buffer(3)]], constant uint *p [[buffer(4)]],
 uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    uint cols=p[0],dtype=p[1],layer=p[2],reads=p[4],threads=p[5];
    threadgroup float parts[32];
    float mean=0;
    if(layer) {
        for(uint block=tid*reads;block<cols;block+=threads*reads) for(uint i=0;i<reads && block+i<cols;i++) mean+=x[ulong(row)*cols+block+i];
        mean=music_group_sum(mean,parts,tid)/float(cols);
    }
    float variance=0;
    for(uint block=tid*reads;block<cols;block+=threads*reads) for(uint i=0;i<reads && block+i<cols;i++) {
        float a=x[ulong(row)*cols+block+i]-mean; variance+=a*a;
    }
    variance=music_group_sum(variance,parts,tid);
    float inv=metal::precise::rsqrt(variance/float(cols)+as_type<float>(p[6]));
    for(uint block=tid*reads;block<cols;block+=threads*reads) for(uint i=0;i<reads && block+i<cols;i++) {
        uint d=block+i; float a=music_round((x[ulong(row)*cols+d]-mean)*inv,dtype);
        a=music_round(a*w[d]+(p[3] ? bias[d] : 0.0f),dtype);
        y[ulong(row)*cols+d]=a;
    }
}
kernel void music3_rope(device const float *x [[buffer(0)]], device float *y [[buffer(1)]], constant uint *p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint dim=p[2],hdim=dim/2,d=gid%hdim,vector=gid/hdim,t=(vector/p[1])%p[0];
    ulong base=ulong(vector)*dim;
    float freq=exp2(-float(d)/float(hdim)*as_type<float>(p[4]));
    float angle=float(t+p[3])*freq, co=metal::fast::cos(angle), si=metal::fast::sin(angle);
    float a=x[base+d], b=x[base+hdim+d];
    // Pin contraction order: cancellation can cross a BF16 rounding tie.
    y[base+d]=music_round(fma(a,co,-b*si),p[5]);
    y[base+hdim+d]=music_round(fma(a,si,b*co),p[5]);
}
kernel void music3_rotary_tables(device const float *unused [[buffer(0)]],device float *y [[buffer(1)]],constant uint *p [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
    uint half_dim=p[1]/2,count=p[0]*half_dim;
    if(gid>=count) return;
    float inv=1.0f/metal::precise::pow(as_type<float>(p[2]),float(gid%half_dim)/float(half_dim));
    float angle=float(gid/half_dim)*inv;
    y[gid]=metal::precise::cos(angle);y[count+gid]=metal::precise::sin(angle);
}
// MLX 0.32.3 erf.h polynomial, Copyright 2023-2024 Apple Inc., MIT licensed.
inline float music_erfinv(float a) {
    float t=log(fma(a,-a,1.0f)),v;
    if(abs(t)>6.125f) {
        v=3.03697567e-10f; v=fma(v,t,2.93243101e-8f); v=fma(v,t,1.22150334e-6f);
        v=fma(v,t,2.84108955e-5f); v=fma(v,t,3.93552968e-4f); v=fma(v,t,3.02698812e-3f);
        v=fma(v,t,4.83185798e-3f); v=fma(v,t,-2.64646143e-1f); v=fma(v,t,8.40016484e-1f);
    } else {
        v=5.43877832e-9f; v=fma(v,t,1.43285448e-7f); v=fma(v,t,1.22774793e-6f);
        v=fma(v,t,1.12963626e-7f); v=fma(v,t,-5.61530760e-5f); v=fma(v,t,-1.47697632e-4f);
        v=fma(v,t,2.31468678e-3f); v=fma(v,t,1.15392581e-2f); v=fma(v,t,-2.32015476e-1f);
        v=fma(v,t,8.86226892e-1f);
    }
    return a*v;
}
kernel void music3_normal(device const float *x [[buffer(0)]], device float *y [[buffer(1)]], constant uint *p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if(gid<p[0]) y[gid]=music_round(music_erfinv(x[gid])*1.4142135623730951f,p[1]);
}
// MLX's decode attention reduces 32 independent online-softmax partitions.
// MLX sdpa_vector.h, Copyright 2024 Apple Inc., MIT licensed.
kernel void music3_attention_vector(device const float *q [[buffer(0)]],
 device const float *k [[buffer(1)]],device const float *v [[buffer(2)]],
 device float *y [[buffer(3)]],constant uint *p [[buffer(4)]],
 uint tid [[thread_index_in_threadgroup]],uint gid [[threadgroup_position_in_grid]]) {
    uint batch=p[0],qlen=p[1],klen=p[2],heads=p[3],kh=p[4],dim=p[5],
         h=gid%heads,t=(gid/heads)%qlen,b=gid/(heads*qlen),kvh=h/(heads/kh),lane=tid%32,sg=tid/32,n=dim/32;
    ulong qi=(ulong(b)*qlen*heads+t*heads+h)*dim;
    float query[8],out[8];
    for(uint i=0;i<n;i++) { query[i]=as_type<float>(p[11])*q[qi+lane*n+i]; out[i]=0; }
    float maximum=-MAXFLOAT,denom=0;
    uint count=p[7] ? min(klen,t+p[8]+1) : klen;
    for(uint s=sg;s<count;s+=32) {
        ulong ki=p[6] ? (ulong(s)*batch*kh+b*kh+kvh)*dim : (ulong(b)*klen*kh+s*kh+kvh)*dim;
        float score=0; for(uint i=0;i<n;i++) score+=query[i]*k[ki+lane*n+i]; score=simd_sum(score);
        float next=max(maximum,score),factor=fast::exp(maximum-next),prob=fast::exp(score-next);
        denom=denom*factor+prob; maximum=next;
        for(uint i=0;i<n;i++) out[i]=out[i]*factor+prob*v[ki+lane*n+i];
    }
    threadgroup float maxima[32],sums[32],outputs[1024];
    if(lane==0) { maxima[sg]=maximum; sums[sg]=denom; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=maxima[lane];float global=simd_max(maximum),factor=fast::exp(maximum-global);
    denom=simd_sum(sums[lane]*factor);
    for(uint i=0;i<n;i++) {
        outputs[lane*32+sg]=out[i];threadgroup_barrier(mem_flags::mem_threadgroup);
        float result=simd_sum(outputs[sg*32+lane]*factor);
        if(lane==0) y[qi+sg*n+i]=music_round(denom==0 ? result : result/denom,p[10]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// An 8x8 SIMD matrix tile preserves MLX's MMA accumulation order.
kernel void music3_linear_mma(device const uchar *w [[buffer(0)]],
 device const float *sc [[buffer(1)]],device const float *off [[buffer(2)]],
 device const uchar *bs [[buffer(3)]],device const float *x [[buffer(4)]],
 device const float *bias [[buffer(5)]],device float *y [[buffer(6)]],
 constant uint *p [[buffer(7)]],uint tid [[thread_index_in_threadgroup]],uint gid [[threadgroup_position_in_grid]]) {
    uint cols=p[1],out=p[4],rows=p[7],nblocks=(out+7)/8,r0=(gid/nblocks)*8,o0=(gid%nblocks)*8;
    threadgroup float xa[64],wb[64],results[64];
    float total[2]={0,0};
    for(uint part=0;part<p[8];part++) {
        simdgroup_float8x8 a,b,c(0.0f);
        uint begin=part*(cols/p[8]),end=begin+cols/p[8];
        for(uint k0=begin;k0<end;k0+=8) {
            for(uint j=0;j<2;j++) {
                uint idx=tid+j*32,r=idx/8,k=idx%8;
                xa[idx]=r0+r<rows && k0+k<end ? x[ulong(r0+r)*cols+k0+k] : 0;
                uint o=o0+idx%8,kk=k0+idx/8;
                float weight=o<out && kk<end ? music_value(w,sc,off,bs,p,o,kk) : 0;
                wb[idx]=p[0]>=3 ? music_round(weight,p[6]) : weight;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(a,xa,8);simdgroup_load(b,wb,8);
            simdgroup_multiply_accumulate(c,a,b,c);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(c,results,8);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<2;j++) total[j]=p[8]>1 ? music_round(total[j]+music_round(results[tid+j*32],p[6]),p[6]) : results[tid+j*32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<2;j++) {
        uint idx=tid+j*32,r=r0+idx/8,o=o0+idx%8;
        if(r<rows && o<out) {
            float value=p[0]>=3 ? music_round(total[j],p[6]) : total[j];
            y[ulong(r)*out+o]=music_round(value+(p[5] ? bias[o] : 0),p[6]);
        }
    }
}
// Tiled form of music3_linear_mma. Every output element still accumulates
// the same sequence of 8x8x8 SIMD-matrix products in K order (split-K parts
// folded exactly as before), so results are bit-identical; what changes is
// reuse. A threadgroup of four simdgroups stages a (rows x 32) activation
// tile and decodes a (32 x outputs) weight tile once, and each simdgroup
// holds 2x2 accumulators. The 8x8 kernel decoded every weight once per 8
// rows and paid two barriers per 8-wide K step. SRxSO must be 4.
template <uint SR, uint SO, uint RM, uint RN>
kernel void music3_linear_tiled(device const uchar *w [[buffer(0)]],
 device const float *sc [[buffer(1)]],device const float *off [[buffer(2)]],
 device const uchar *bs [[buffer(3)]],device const float *x [[buffer(4)]],
 device const float *bias [[buffer(5)]],device float *y [[buffer(6)]],
 constant uint *p [[buffer(7)]],uint tid [[thread_index_in_threadgroup]],uint gid [[threadgroup_position_in_grid]]) {
    // Each simdgroup owns (8*RM) rows by (8*RN) outputs of the tile.
    constexpr uint SGR=8*RM,SGO=8*RN,SGE=SGR*SGO,PER=SGE/32,TR=SR*SGR,TO=SO*SGO,BK=32;
    uint cols=p[1],out=p[4],rows=p[7],parts=p[8];
    uint oblocks=(out+TO-1)/TO,r0=(gid/oblocks)*TR,o0=(gid%oblocks)*TO;
    uint simd=tid/32,lane=tid%32,sr=simd/SO,so=simd%SO;
    threadgroup float xs[TR*BK],ws[BK*TO],res[SR*SO*SGE];
    float total[PER];
    for(uint m=0;m<PER;m++) total[m]=0;
    // Quantized split-K: MLX's qmm_t_splitk writes each partial in the activation
    // dtype and sums them with its column reduce. Below 32 partials that kernel
    // gives each of min(8, parts) threads every 8th row, summed in bf16, then folds
    // the thread totals in order; at 32 its looped kernel ends in a simd_sum, a
    // float sum rounded once. `lanes` holds those thread totals. (Kokoro, p[12],
    // keeps its own sequential fold.)
    bool mlx_reduce=!p[12];
    float lanes[8][PER];
    for(uint t=0;t<8;t++) for(uint m=0;m<PER;m++) lanes[t][m]=0;
    bool quant=p[0]>=3,fast4=p[0]==3 && p[2]==4;
    uint words=(cols*p[2]+31)/32;
    for(uint part=0;part<parts;part++) {
        simdgroup_float8x8 c[RM][RN];
        for(uint i=0;i<RM;i++) for(uint j=0;j<RN;j++) c[i][j]=simdgroup_float8x8(0.0f);
        uint begin=part*(cols/parts),end=begin+cols/parts;
#ifdef KOKORO_CONTRACTS
        if(p[12] && parts>1) {
            uint partition=(cols/16/parts)*16;
            begin=part*partition;end=part+1==parts ? cols : begin+partition;
        }
#endif
        for(uint k0=begin;k0<end;k0+=BK) {
            if((cols&3)==0 && (begin&3)==0 && k0+BK<=end) {
                // Full K chunk with 4-aligned rows: four-wide loads (packed, so
                // no alignment is assumed beyond a float). Same values as below.
                for(uint idx=tid;idx<TR*(BK/4);idx+=128) {
                    uint r=idx/(BK/4),kc=(idx%(BK/4))*4;
                    float4 v=0;
                    if(r0+r<rows) v=float4(*((device const packed_float4*)(x+ulong(r0+r)*cols+k0+kc)));
                    xs[r*BK+kc]=v.x;xs[r*BK+kc+1]=v.y;xs[r*BK+kc+2]=v.z;xs[r*BK+kc+3]=v.w;
                }
            } else {
                for(uint idx=tid;idx<TR*BK;idx+=128) {
                    uint r=idx/BK,kc=idx%BK;
                    xs[idx]=r0+r<rows && k0+kc<end ? x[ulong(r0+r)*cols+k0+kc] : 0;
                }
            }
            if(fast4) {
                // Eight consecutive four-bit codes share one 32-bit word and
                // one scale/offset (group % 8 == 0); same per-element value as
                // music_value. Consecutive lanes take consecutive outputs. The
                // word, scale and offset are read once per lane and reused for
                // all eight codes; each weight is still float(code)*scale+offset.
                uint gpr=cols/p[3];
                for(uint idx=tid;idx<TO*(BK/8);idx+=128) {
                    uint oo=idx%TO,q=idx/TO,o=o0+oo,kk=k0+q*8;
                    bool ok=o<out && kk<end;
                    uint word=0;float sv=0,ov=0;
                    if(ok) {
                        word=((device const uint*)w)[ulong(o)*words+kk/8];
                        ulong g=ulong(o)*gpr+kk/p[3];
                        sv=music_sc(sc,g);ov=music_sc(off,g);
                    }
                    for(uint i=0;i<8;i++) {
                        float weight=ok ? float((word>>(4*i))&15u)*sv+ov : 0.0f;
                        ws[(q*8+i)*TO+oo]=music_round(weight,p[6]);
                    }
                }
            } else {
                for(uint idx=tid;idx<TO*BK;idx+=128) {
                    uint oo=idx/BK,kc=idx%BK,o=o0+oo,kk=k0+kc;
                    float weight=0;
                    if(o<out && kk<end) weight=music_value(w,sc,off,bs,p,o,kk);
                    ws[kc*TO+oo]=quant ? music_round(weight,p[6]) : weight;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint kb=0;kb<BK/8;kb++) {
                if(k0+kb*8>=end) break;
                simdgroup_float8x8 a[RM],b[RN];
                for(uint i=0;i<RM;i++) simdgroup_load(a[i],xs+(sr*SGR+i*8)*BK+kb*8,BK);
                for(uint j=0;j<RN;j++) simdgroup_load(b[j],ws+(kb*8)*TO+so*SGO+j*8,TO);
                for(uint i=0;i<RM;i++) for(uint j=0;j<RN;j++) simdgroup_multiply_accumulate(c[i][j],a[i],b[j],c[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint i=0;i<RM;i++) for(uint j=0;j<RN;j++) simdgroup_store(c[i][j],res+simd*SGE+i*8*SGO+j*8,SGO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint m=0;m<PER;m++) {
            float r=res[simd*SGE+lane+32*m];
            // p[13]: dense split-K sums float32 partials in order (MLX's gemm_splitk_accum);
            // the quantized path stores each partial in the activation dtype first.
            if(parts==1) total[m]=r;
            else if(p[13]!=0) total[m]+=r;
            else if(mlx_reduce) {
                float rb=music_round(r,p[6]);
                if(parts==32) lanes[0][m]+=rb;
                else { uint L=min(8u,parts),t=part%L; lanes[t][m]=part<L ? rb : music_round(lanes[t][m]+rb,p[6]); }
            }
            else total[m]=music_round(total[m]+music_round(r,p[6]),p[6]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(parts>1 && p[13]==0 && mlx_reduce) {
        for(uint m=0;m<PER;m++) {
            if(parts==32) total[m]=music_round(lanes[0][m],p[6]);
            else {
                uint L=min(8u,parts);
                float tot=lanes[0][m];
                for(uint t=1;t<L;t++) tot=music_round(lanes[t][m]+tot,p[6]);
                total[m]=tot;
            }
        }
    }
    for(uint m=0;m<PER;m++) {
        uint e=lane+32*m,row=r0+sr*SGR+e/SGO,o=o0+so*SGO+e%SGO;
        if(row<rows && o<out) {
            float value=quant ? music_round(total[m],p[6]) : total[m];
            y[ulong(row)*out+o]=music_round(value+(p[5] ? bias[o] : 0),p[6]);
        }
    }
}
template [[host_name("music3_linear_tiled_32x32")]] [[kernel]] decltype(music3_linear_tiled<2,2,2,2>) music3_linear_tiled<2,2,2,2>;
template [[host_name("music3_linear_tiled_16x64")]] [[kernel]] decltype(music3_linear_tiled<1,4,2,2>) music3_linear_tiled<1,4,2,2>;
kernel void music3_snake(device const float *x [[buffer(0)]],device const float *alpha [[buffer(1)]],device float *y [[buffer(2)]],constant uint *p [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
    if(gid>=p[2]) return;
    float a=alpha[gid/p[0]],s=music_round(metal::precise::sin(music_round(a*x[gid],p[1])),p[1]);
    float square=music_round(metal::precise::pow(s,as_type<float>(p[3])),p[1]);
    float denom=music_round(a+music_round(1e-9f,p[1]),p[1]);
    y[gid]=music_round(x[gid]+music_round(square/denom,p[1]),p[1]);
}
// Full attention uses FP32 MMA, exp2 softmax, and BK-sized online updates.
kernel void music3_attention_mma(device const float *q [[buffer(0)]],device const float *k [[buffer(1)]],device const float *v [[buffer(2)]],device float *y [[buffer(3)]],constant uint *p [[buffer(4)]],uint lane [[thread_index_in_threadgroup]],uint gid [[threadgroup_position_in_grid]]) {
    uint batch=p[0],qlen=p[1],klen=p[2],heads=p[3],kh=p[4],dim=p[5],blocks=(qlen+7)/8,
        r0=(gid%blocks)*8,h=(gid/blocks)%heads,b=gid/(blocks*heads),kvh=h/(heads/kh),bk=dim>=128 ? 16 : 32;
    uint qid=lane/4,fr=(qid&4)+((lane/2)%4),fc=(qid&2)*2+(lane%2)*2;
    threadgroup float a[64],bmat[64],temp[64],scores[256],maxima[8],sums[8],factors[8];
    simdgroup_float8x8 outputs[32];
    for(uint tile=0;tile<dim/8;tile++) outputs[tile]=simdgroup_float8x8(0.0f);
    if(lane<8) {maxima[lane]=-MAXFLOAT;sums[lane]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint start=0;start<klen;start+=bk) {
        for(uint tile=0;tile<bk/8;tile++) {
            simdgroup_float8x8 am,bm,cm(0.0f);
            for(uint d0=0;d0<dim;d0+=8) {
                for(uint j=0;j<2;j++) {
                    uint idx=lane+j*32,r=r0+idx/8,d=d0+idx%8;
                    a[idx]=r<qlen ? q[((ulong(b)*qlen+r)*heads+h)*dim+d] : 0;
                    uint key=start+tile*8+idx%8,dd=d0+idx/8;
                    ulong ki=p[6] ? (ulong(key)*batch*kh+b*kh+kvh)*dim : (ulong(b)*klen*kh+key*kh+kvh)*dim;
                    bmat[idx]=key<klen ? k[ki+dd] : 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(am,a,8);simdgroup_load(bm,bmat,8);simdgroup_multiply_accumulate(cm,am,bm,cm);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(cm,temp,8);threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint j=0;j<2;j++) {
                uint idx=lane+j*32,r=idx/8,key=start+tile*8+idx%8;
                scores[r*bk+tile*8+idx%8]=key<klen && (!p[7] || key<=r0+r+p[8]) ? temp[idx]*(as_type<float>(p[11])*1.4426950408889634f) : -MAXFLOAT;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float maximum=maxima[fr];
        for(uint tile=0;tile<bk/8;tile++) {
            float val=max(scores[fr*bk+tile*8+fc],scores[fr*bk+tile*8+fc+1]);
            val=max(val,simd_shuffle_xor(val,ushort(1)));val=max(val,simd_shuffle_xor(val,ushort(8)));maximum=max(maximum,val);
        }
        float factor=fast::exp2(maxima[fr]-maximum),total=0;
        for(uint tile=0;tile<bk/8;tile++) {
            uint idx=fr*bk+tile*8+fc;
            float x0=fast::exp2(scores[idx]-maximum),x1=fast::exp2(scores[idx+1]-maximum);
            scores[idx]=x0;scores[idx+1]=x1;
            float val=x0+x1;val+=simd_shuffle_xor(val,ushort(1));val+=simd_shuffle_xor(val,ushort(8));total+=val;
        }
        if(fc==0) {sums[fr]=sums[fr]*factor+total;maxima[fr]=maximum;factors[fr]=factor;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint tile=0;tile<dim/8;tile++) {
            simdgroup_store(outputs[tile],temp,8);threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint j=0;j<2;j++) {uint idx=lane+j*32;temp[idx]*=factors[idx/8];}
            threadgroup_barrier(mem_flags::mem_threadgroup);simdgroup_load(outputs[tile],temp,8);
            for(uint kk=0;kk<bk/8;kk++) {
                for(uint j=0;j<2;j++) {
                    uint idx=lane+j*32;a[idx]=scores[(idx/8)*bk+kk*8+idx%8];
                    uint key=start+kk*8+idx/8,d=tile*8+idx%8;
                    ulong ki=p[6] ? (ulong(key)*batch*kh+b*kh+kvh)*dim : (ulong(b)*klen*kh+key*kh+kvh)*dim;
                    bmat[idx]=key<klen ? v[ki+d] : 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);simdgroup_float8x8 am,bm;simdgroup_load(am,a,8);simdgroup_load(bm,bmat,8);
                simdgroup_multiply_accumulate(outputs[tile],am,bm,outputs[tile]);threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    for(uint tile=0;tile<dim/8;tile++) {
        simdgroup_store(outputs[tile],temp,8);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<2;j++) {uint idx=lane+j*32,r=r0+idx/8;if(r<qlen) y[((ulong(b)*qlen+r)*heads+h)*dim+tile*8+idx%8]=music_round(temp[idx]/sums[idx/8],p[10]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
