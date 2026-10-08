// Port-local shared audio additions. The pinned Kokoro MLX LSTM gate order
// and audio::ops grouped convolution define layout; tests use independent MLX.
#include <metal_stdlib>
using namespace metal;

kernel void audio_conv_grouped(device const float *w [[buffer(0)]], device const float *x [[buffer(1)]],
 device const float *bias [[buffer(2)]], device float *y [[buffer(3)]], constant uint *p [[buffer(4)]],
 uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint ic=p[0],oc=p[1],ks=p[2],stride=p[3],pad=p[4],dilation=p[5],transpose=p[6],len=p[7],outlen=p[8],groups=p[12];
    uint o=gid/outlen,t=gid%outlen,ig=ic/groups,og=oc/groups,first=(o/og)*ig;
    float sum=0;
    for(uint f=tid;f<ig*ks;f+=128) {
        uint i=first+f/ks,k=f%ks;
        int position=int(t*stride+k*dilation)-int(pad);
        if(transpose) {position=int(t+pad)-int(k); if(position<0 || uint(position)%stride!=0) continue; position/=int(stride);}
        if(position>=0 && uint(position)<len) {
            ulong wi=transpose ? (ulong(i)*og+(o%og))*ks+k : (ulong(o)*ig+(i-first))*ks+k;
            sum+=x[ulong(i)*len+uint(position)]*w[wi];
        }
    }
    sum=simd_sum(sum);
    threadgroup float parts[4];
    if(tid%32==0) parts[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) y[gid]=(parts[0]+parts[1]+parts[2]+parts[3])+(p[9] ? bias[o] : 0.0f);
}

// MLX 0.31.2 GEMV order and precise unaries are part of Kokoro F32 recurrence.
inline float audio_sigmoid(float x) { float y=1.0f/(1.0f+exp(abs(x)));return x<0 ? y : 1.0f-y; }
inline float audio_recurrent_dot(device const float *w,threadgroup const float *hidden,uint width,uint row,uint lane) {
 float sum=0;
 for(uint k=lane*4;k<width;k+=128)for(uint i=0;i<4 && k+i<width;i++)sum+=hidden[k+i]*w[ulong(row)*width+k+i];
 sum+=simd_shuffle_down(sum,ushort(16));sum+=simd_shuffle_down(sum,ushort(8));
 sum+=simd_shuffle_down(sum,ushort(4));sum+=simd_shuffle_down(sum,ushort(2));sum+=simd_shuffle_down(sum,ushort(1));return sum;
}
#pragma clang fp contract(off)
kernel void audio_lstm_recurrence(device const float *w [[buffer(0)]], device const float *projection [[buffer(1)]],
 device float *out [[buffer(2)]], constant uint *p [[buffer(3)]], uint tid [[thread_index_in_threadgroup]],uint threads [[threads_per_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 uint seq=p[0],width=p[1];threadgroup float hidden[512],cell[512],gatesall[2048];
 if(tid<width) {hidden[tid]=0;cell[tid]=0;}threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint step=0;step<seq;step++) {
  uint t=p[2] ? seq-1-step : step;
  for(uint d=simd;d<width;d+=threads/32)for(uint g=0;g<4;g++){
   float dot=audio_recurrent_dot(w,hidden,width,g*width+d,lane);
   if(lane==0)gatesall[g*width+d]=dot+projection[(ulong(t)*4+g)*width+d];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float next=0,new_cell=0;
  if(tid<width){float i=audio_sigmoid(gatesall[tid]),f=audio_sigmoid(gatesall[width+tid]),g=metal::precise::tanh(gatesall[2*width+tid]),o=audio_sigmoid(gatesall[3*width+tid]);new_cell=f*cell[tid]+i*g;next=o*metal::precise::tanh(new_cell);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid<width){hidden[tid]=next;cell[tid]=new_cell;out[ulong(t)*width+tid]=next;}threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
#pragma clang fp contract(on)

// Scan ordering and radix4/radix5 codelets adapted from MLX v0.31.2.
// Copyright (c) 2023-2024 Apple Inc. MIT license follows.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
kernel void audio_cumsum(device const float *x [[buffer(0)]],device const float *unused [[buffer(1)]],
 device float *out [[buffer(2)]],constant uint *p [[buffer(3)]],uint column [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    uint lane=tid%32,simd=tid/32;
    threadgroup float sums[32];
    float prefix=0;
    for(uint first=0;first<p[0];first+=4*p[2]) {
        float values[4];
        for(uint i=0;i<4;i++) {uint row=first+4*tid+i;values[i]=row<p[0] ? x[ulong(row)*p[1]+column] : 0;}
        for(uint i=1;i<4;i++) values[i]=values[i]+values[i-1];
        float prev_thread=simd_prefix_exclusive_sum(values[3]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<32) sums[tid]=0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane==31) sums[simd]=prev_thread+values[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(simd==0) {float prev_group=simd_prefix_exclusive_sum(sums[lane]);sums[lane]=prev_group;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0;i<4;i++) {
            values[i]=values[i]+prefix;values[i]=values[i]+sums[simd];values[i]=values[i]+prev_thread;
            uint row=first+4*tid+i;if(row<p[0]) out[ulong(row)*p[1]+column]=values[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==p[2]-1) sums[0]=values[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix=sums[0];
    }
}
inline float2 audio_complex_mul(float2 a,float2 b) {return float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
inline void audio_radix4(thread float2 *x,thread float2 *y) {
    float2 z0=x[0]+x[2],z1=x[0]-x[2],z2=x[1]+x[3],z3=x[1]-x[3];
    float2 z3i={z3.y,-z3.x};
    y[0]=z0+z2;y[1]=z1+z3i;y[2]=z0-z2;y[3]=z1-z3i;
}
inline void audio_radix5(thread float2 *x,thread float2 *y) {
    float2 a1=x[1]+x[4],a2=x[2]+x[3],a3=x[1]-x[4],a4=x[2]-x[3];
    float2 a5=a1+a2,a6=0.5590169943749475f*(a1-a2),a7=x[0]-a5/4;
    float2 a8=a7+a6,a9=a7-a6,a10=0.9510565162951535f*a3+0.5877852522924731f*a4,a11=0.5877852522924731f*a3-0.9510565162951535f*a4;
    float2 a10j={a10.y,-a10.x},a11j={a11.y,-a11.x};
    y[0]=x[0]+a5;y[1]=a8+a10j;y[2]=a9+a11j;y[3]=a9-a11j;y[4]=a8-a10j;
}
inline float audio_windowed(device const float *signal,device const float *window,uint frame,uint t,uint length,uint hop) {
    int index=int(frame*hop+t)-10;
    if(index<0) index=-index;
    if(index>=int(length)) index=2*int(length)-2-index;
    return signal[index]*window[t];
}
kernel void audio_stft20(device const float *signal [[buffer(0)]],device const float *window [[buffer(1)]],device float *out [[buffer(2)]],
 constant uint *p [[buffer(3)]],uint pair [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    if(lane!=0) return;
    uint frame=pair*2,frames=p[1],next=min(frame+1,frames-1);
    float2 buf[20],stage[20],x[5],y[5];
    for(uint t=0;t<20;t++) buf[t]={audio_windowed(signal,window,frame,t,p[0],p[2]),audio_windowed(signal,window,next,t,p[0],p[2])};
    // FFT20 Stockham decomposition is radix4 followed by radix5.
    for(uint i=0;i<5;i++) {
        for(uint r=0;r<4;r++) x[r]=buf[i+r*5];
        audio_radix4(x,y);
        for(uint r=0;r<4;r++) stage[i*4+r]=y[r];
    }
    for(uint i=0;i<4;i++) {
        for(uint r=0;r<5;r++) x[r]=stage[i+r*4];
        float theta=-2.0f*float(i)*M_PI_F/20;
        float2 first={fast::cos(theta),fast::sin(theta)},twiddle=first;
        x[1]=audio_complex_mul(x[1],twiddle);
        for(uint r=2;r<5;r++) {twiddle=audio_complex_mul(twiddle,first);x[r]=audio_complex_mul(x[r],twiddle);}
        audio_radix5(x,y);
        for(uint r=0;r<5;r++) buf[i+r*4]=y[r];
    }
    for(uint b=0;b<11;b++) {
        float2 a,z;
        if(b==0) {a={buf[0].x,0};z={buf[0].y,0};}
        else {float2 partner=buf[20-b]*float2(1,-1);a=(buf[b]+partner)/2;z=audio_complex_mul((buf[b]-partner)/2,float2(0,-1));}
        out[ulong(b)*frames+frame]=sqrt(a.x*a.x+a.y*a.y);
        out[ulong(11+b)*frames+frame]=atan2(a.y,a.x);
        if(frame+1<frames) {
            out[ulong(b)*frames+frame+1]=sqrt(z.x*z.x+z.y*z.y);
            out[ulong(11+b)*frames+frame+1]=atan2(z.y,z.x);
        }
    }
}

// Materialized Kokoro attention, with pinned MLX 0.31.2 row softmax. All buffers store
// logical F32/F16/BF16 values as floats; music_round applies the native casts.
// p: batch, queries, keys, heads, kv_heads, dim, time_major, causal, offset,
// threads, dtype, scale_bits. Q/output are BQHD; K/V are BKHD or KBHD.
kernel void audio_bert_qk(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint gid [[threadgroup_position_in_grid]]) {
    uint qtiles=(p[1]+7)/8, ktiles=(p[2]+7)/8;
    uint k0=(gid%ktiles)*8, q0=((gid/ktiles)%qtiles)*8;
    uint h=(gid/(ktiles*qtiles))%p[3], b=gid/(ktiles*qtiles*p[3]);
    uint kvh=h/(p[3]/p[4]), dim=p[5];
    threadgroup float a[64], bmat[64], result[64];
    simdgroup_float8x8 am, bm, acc(0.0f);
    for(uint d0=0;d0<dim;d0+=8) {
        for(uint j=0;j<2;j++) {
            uint idx=lane+j*32, query=q0+idx/8, d=d0+idx%8;
            a[idx]=query<p[1] && d<dim
                ? q[((ulong(b)*p[1]+query)*p[3]+h)*dim+d] : 0.0f;
            uint key=k0+idx%8, kd=d0+idx/8;
            ulong base=p[6] ? (ulong(key)*p[0]*p[4]+b*p[4]+kvh)*dim
                : (ulong(b)*p[2]*p[4]+key*p[4]+kvh)*dim;
            bmat[idx]=key<p[2] && kd<dim ? k[base+kd] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(am,a,8);
        simdgroup_load(bm,bmat,8);
        simdgroup_multiply_accumulate(acc,am,bm,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,result,8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<2;j++) {
        uint idx=lane+j*32, query=q0+idx/8, key=k0+idx%8;
        if(query<p[1] && key<p[2]) {
            ulong out=((ulong(b)*p[3]+h)*p[1]+query)*p[2]+key;
            scores[out]=p[7] && key>query+p[8] ? -INFINITY
                : result[idx]/as_type<float>(p[11]);
        }
    }
}

// Four reads per thread and the two SIMD reductions preserve MLX's FP32
// normalizer before the reciprocal multiply and final logical dtype cast.
kernel void audio_bert_softmax(
    device const float *scores [[buffer(0)]],
    device float *probabilities [[buffer(1)]],
    constant uint *p [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]) {
    uint keys=p[2];
    ulong base=ulong(gid)*keys;
    threadgroup float local_max[32], local_normalizer[32];
    if(group==0) {
        local_max[lane]=-INFINITY;
        local_normalizer[lane]=0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(keys<=4096) {
        float values[4], maximum=-MAXFLOAT;
        for(uint i=0;i<4;i++) {
            uint key=tid*4+i;
            values[i]=key<keys ? scores[base+key] : -INFINITY;
            maximum=maximum<values[i] ? values[i] : maximum;
        }
        maximum=simd_max(maximum);
        if(lane==0) local_max[group]=maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(group==0) {
            maximum=simd_max(local_max[lane]);
            if(lane==0) local_max[0]=maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum=local_max[0];
        float normalizer=0.0f;
        for(uint i=0;i<4;i++) {
            values[i]=fast::exp(values[i]-maximum);
            normalizer+=values[i];
        }
        normalizer=simd_sum(normalizer);
        if(lane==0) local_normalizer[group]=normalizer;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(group==0) {
            normalizer=simd_sum(local_normalizer[lane]);
            if(lane==0) local_normalizer[0]=normalizer;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        normalizer=1.0f/local_normalizer[0];
        for(uint i=0;i<4;i++) {
            uint key=tid*4+i;
            if(key<keys) probabilities[base+key]=values[i]*normalizer;
        }
    } else {
        uint rounds=(keys+4*threads-1)/(4*threads);
        float maximum=-MAXFLOAT, normalizer=0.0f, previous;
        for(uint r=0;r<rounds;r++) {
            uint offset=r*threads*4+tid*4;
            float values[4];
            for(uint i=0;i<4;i++) values[i]=offset+i<keys ? scores[base+offset+i] : -INFINITY;
            previous=maximum;
            for(uint i=0;i<4;i++) maximum=maximum<values[i] ? values[i] : maximum;
            normalizer*=fast::exp(previous-maximum);
            for(uint i=0;i<4;i++) normalizer+=fast::exp(values[i]-maximum);
        }
        previous=maximum;
        maximum=simd_max(maximum);
        normalizer*=fast::exp(previous-maximum);
        normalizer=simd_sum(normalizer);
        previous=maximum;
        if(lane==0) local_max[group]=maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum=simd_max(local_max[lane]);
        normalizer*=fast::exp(previous-maximum);
        if(lane==0) local_normalizer[group]=normalizer;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        normalizer=simd_sum(local_normalizer[lane]);
        normalizer=1.0f/normalizer;
        for(uint r=0;r<rounds;r++) {
            uint offset=r*threads*4+tid*4;
            for(uint i=0;i<4;i++) {
                if(offset+i<keys) probabilities[base+offset+i]=music_round(
                    fast::exp(scores[base+offset+i]-maximum)*normalizer,p[10]);
            }
        }
    }
}

kernel void audio_bert_pv(
    device const float *probabilities [[buffer(0)]],
    device const float *v [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint gid [[threadgroup_position_in_grid]]) {
    uint qtiles=(p[1]+7)/8, dtiles=(p[5]+7)/8;
    uint d0=(gid%dtiles)*8, q0=((gid/dtiles)%qtiles)*8;
    uint h=(gid/(dtiles*qtiles))%p[3], b=gid/(dtiles*qtiles*p[3]);
    uint kvh=h/(p[3]/p[4]), dim=p[5];
    threadgroup float a[64], bmat[64], result[64];
    simdgroup_float8x8 am, bm, acc(0.0f);
    for(uint k0=0;k0<p[2];k0+=8) {
        for(uint j=0;j<2;j++) {
            uint idx=lane+j*32, query=q0+idx/8, key=k0+idx%8;
            a[idx]=query<p[1] && key<p[2]
                ? probabilities[((ulong(b)*p[3]+h)*p[1]+query)*p[2]+key] : 0.0f;
            uint vk=k0+idx/8, d=d0+idx%8;
            ulong base=p[6] ? (ulong(vk)*p[0]*p[4]+b*p[4]+kvh)*dim
                : (ulong(b)*p[2]*p[4]+vk*p[4]+kvh)*dim;
            bmat[idx]=vk<p[2] && d<dim ? v[base+d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(am,a,8);
        simdgroup_load(bm,bmat,8);
        simdgroup_multiply_accumulate(acc,am,bm,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,result,8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<2;j++) {
        uint idx=lane+j*32, query=q0+idx/8, d=d0+idx%8;
        if(query<p[1] && d<dim) output[((ulong(b)*p[1]+query)*p[3]+h)*dim+d]
            =result[idx];
    }
}

// erf and expm1 adapted from MLX v0.31.2; MIT and BSD notices retained.
// Copyright (c) 2023 Apple Inc.





// Original license copied below:
//  Copyright (c) 2015-2023 Norbert Juffa
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/* Compute exponential base e minus 1. Maximum ulp error = 0.997458

   i = rint(a/log(2)), f = a-i*log(2). Then expm1(a) = 2**i * (expm1(f)+1) - 1.
   Compute r = expm1(f). Then expm1(a)= 2 * (0.5 * 2**i * r + 0.5 * 2**i - 0.5).
   With t = 0.5*2**i, expm1(a) = 2*(r * t + t-0.5). However, for best accuracy,
   when i == 1, expm1(a)= 2*(r + 0.5), and when i == 0, expm1(a) = r.

   NOTE: Scale factor b is only applied if i < 0 or i > 1 (should be power of 2)
*/
float audio_expm1_scaled(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;

  // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
  j = fma(1.442695f, a, 12582912.f); // 0x1.715476p0, 0x1.8p23
  j = j - 12582912.0f; // 0x1.8p23
  i = (int)j;
  f = fma(j, -6.93145752e-1f, a);

  // approximate r = exp(f)-1 on interval [-log(2)/2, +log(2)/2]
  s = f * f;
  if (a == 0.0f)
    s = a; // ensure -0 is passed through
  // err = 0.997458  ulp1 = 11081805
  r = 1.97350979e-4f; // 0x1.9de000p-13
  r = fma(r, f, 1.39309070e-3f); // 0x1.6d30bcp-10
  r = fma(r, f, 8.33343994e-3f); // 0x1.1111f6p-7
  r = fma(r, f, 4.16668020e-2f); // 0x1.55559ep-5
  r = fma(r, f, 1.66666716e-1f); // 0x1.55555cp-3
  r = fma(r, f, 4.99999970e-1f); // 0x1.fffffep-2
  u = (j == 1) ? (f + 0.5f) : f;
  v = fma(r, s, u);
  s = 0.5f * b;
  t = ldexp(s, i);
  y = t - s;
  x = (t - y) - s; // double-float canonicalization of difference
  r = fma(v, t, x) + y;
  r = r + r;
  if (j == 0)
    r = v;
  if (j == 1)
    r = v + v;
  return r;
}

/* Compute exponential base e minus 1. max ulp err = 0.99746 */
float audio_expm1(float a) {
  float r;

  r = audio_expm1_scaled(a, 1.0f);
  /* handle severe overflow and underflow */
  if (abs(a - 1.0f) > 88.0f) {
    r = pow(2, a);
    r = fma(r, r, -1.0f);
  }
  return r;
}

/*
 * Approximation to the error function.
 * Based on code from:
 * https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff#answer-35148199
 */
float audio_erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    // maximum error 0.99527 ulp
    r = metal::fma(
        -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
    u = metal::fma(
        -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
    r = metal::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
    r = metal::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
    r = metal::fma(r, t, -t);
    r = -audio_expm1(r);
    r = metal::copysign(r, a);
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4f; // -0x1.38e000p-11
    r = metal::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
    r = metal::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
    r = metal::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
    r = metal::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
    r = metal::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
    r = metal::fma(r, a, a);
  }
  return r;
}


kernel void audio_gelu(device const float *x [[buffer(0)]], device const float *unused [[buffer(1)]],
 device float *out [[buffer(2)]], constant uint *p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if(gid<p[0]) {
        float v=x[gid];
        // MLX compiled.h prints F32 constants at seven significant digits.
        // Its sqrt(2) divisor is 1.414214, which differs from eager GELU.
        out[gid]=(v*(1.0f+audio_erf(v/1.414214f)))/2.0f;
    }
}
// Contiguous reduction follows MLX 0.31.2 reduce_row.h. Sanitization makes
// [channel,kernel,input] contiguous before squaring, while checkpoints use OKI.
kernel void audio_weight_norm(device const float *x [[buffer(0)]], device const float *g [[buffer(1)]],
 device float *out [[buffer(2)]], constant uint *p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
    uint input=p[0],taps=p[1],width=input*taps,threads=p[2];
    float total=0.0f;
    if(width<=64) {
        if(tid==0) for(uint n=0;n<width;n++) {
            float v=x[ulong(row)*width+(n%input)*taps+n/input];
            float square=v*v; total=square+total;
        }
    } else {
        for(uint first=tid*4;first<width;first+=threads*4) for(uint n=first;n<min(first+4,width);n++) {
            float v=x[ulong(row)*width+(n%input)*taps+n/input];
            float square=v*v; total=square+total;
        }
        total=simd_sum(total);
    }
    threadgroup float parts[32];
    if(tid<32) parts[tid]=0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid%32==0) parts[tid/32]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(threads>32) total=simd_sum(tid<threads/32 ? parts[tid] : 0.0f);
    if(tid==0) parts[0]=sqrt(total)+1e-7f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=parts[0];
    for(uint n=tid;n<width;n+=threads) {
        ulong i=ulong(row)*width+n;
        out[i]=(x[i]/denominator)*g[row];
    }
}

// MLX 0.31.2 reduction contracts, Apple MIT license above applies.
// Layout is explicit: the predictor mean precedes channel-major materialization.
#pragma clang fp contract(off)
kernel void audio_normalization_mean(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
 constant uint *p [[buffer(2)]],uint rowpart [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint cols=p[1],parts=p[5],row=rowpart/parts,part=rowpart%parts; x+=ulong(row)*cols;
 threadgroup float shared[32];float sum=0;
 if(p[3]) {
  if(p[7]) {
   if(tid<p[6])for(uint k=part*p[6]+tid;k<cols;k+=parts*p[6])sum+=x[k];
   shared[tid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
   if(tid==0){sum=shared[0];for(uint i=1;i<p[6];i++)sum+=shared[i];}
  } else if(cols<32) {
   if(tid<8)for(uint k=tid;k<cols;k+=min(cols,8u))sum+=x[k];
   if(tid<8)shared[tid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
   if(tid==0){sum=shared[0];for(uint i=1;i<min(cols,8u);i++)sum+=shared[i];}
  } else {
   for(uint k=part*32+tid;k<cols;k+=32*parts)sum+=x[k];
   sum=simd_sum(sum);
  }
 } else if(cols<=64) { if(tid==0)for(uint k=0;k<cols;k++)sum+=x[k]; }
 else {
  for(uint k=tid*4;k<cols;k+=p[4]*4)for(uint i=0;i<4 && k+i<cols;i++)sum+=x[k+i];
  sum=music_group_sum(sum,shared,tid);
 }
 if(tid==0)out[rowpart]=parts==1 ? sum*(1.0f/float(cols)) : sum;
}
kernel void audio_normalization_output(device const float *x [[buffer(0)]],device const float *means [[buffer(1)]],device float *out [[buffer(2)]],
 constant uint *p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint cols=p[1],threads=p[4],parts=p[5];x+=ulong(row)*cols;out+=ulong(row)*cols;
 threadgroup float shared[32];float mean=means[row];
 if(parts>1){mean=0;for(uint i=tid%32;i<parts;i+=32)mean+=means[row*parts+i];mean=simd_sum(mean)*(1.0f/float(cols));}
 float sum=0;
 if(cols<=64){if(tid==0)for(uint k=0;k<cols;k++){float d=x[k]-mean;float squared=d*d;sum+=squared;}if(tid==0)shared[0]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);sum=shared[0];}
 else {
  for(uint k=tid*4;k<cols;k+=threads*4)for(uint i=0;i<4 && k+i<cols;i++){float d=x[k+i]-mean;float squared=d*d;sum+=squared;}
  sum=music_group_sum(sum,shared,tid);
 }
 float std=sqrt(sum*(1.0f/float(cols))+as_type<float>(p[2]));
 for(uint k=tid*4;k<cols;k+=threads*4)for(uint i=0;i<4 && k+i<cols;i++)out[k+i]=(x[k+i]-mean)/std;
}
#pragma clang fp contract(on)
// Depthwise transpose uses ascending tap order in full F32 MMA tiles.
kernel void audio_depthwise_transpose(device const float *w [[buffer(0)]],device const float *x [[buffer(1)]],device const float *bias [[buffer(2)]],device float *y [[buffer(3)]],constant uint *p [[buffer(4)]],uint lane [[thread_index_in_threadgroup]],uint gid [[threadgroup_position_in_grid]]) {
 uint tiles=(p[8]+7)/8,ch=gid/tiles,t0=(gid%tiles)*8;threadgroup float a[64],b[64],result[64];simdgroup_float8x8 am,bm,acc(0.0f);
 for(uint k0=0;k0<p[2];k0+=8) {
  for(uint j=0;j<2;j++) {
   uint idx=lane+32*j,t=t0+idx/8,k=k0+idx%8;int pos=int(t+p[4])-int(k);bool valid=k<p[2] && t<p[8] && pos>=0 && uint(pos)%p[3]==0;pos/=int(p[3]);a[idx]=valid && uint(pos)<p[7]?x[ulong(ch)*p[7]+uint(pos)]:0;
   uint kk=k0+idx/8;b[idx]=idx%8==0 && kk<p[2]?w[ulong(ch)*p[2]+kk]:0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);simdgroup_load(am,a,8);simdgroup_load(bm,b,8);simdgroup_multiply_accumulate(acc,am,bm,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(acc,result,8);threadgroup_barrier(mem_flags::mem_threadgroup);
 if(lane<8 && t0+lane<p[8])y[ulong(ch)*p[8]+t0+lane]=result[lane*8]+(p[9]?bias[ch]:0.0f);
}
kernel void audio_source_unary(device const float *x [[buffer(0)]],device const float *unused [[buffer(1)]],device float *y [[buffer(2)]],constant uint *p [[buffer(3)]],uint gid [[thread_position_in_grid]]) {if(gid<p[0])y[gid]=p[1]==0?metal::precise::sin(x[gid]):metal::precise::tanh(x[gid]);}
// The single-output projection is a matrix-vector product over input rows.
kernel void audio_vector_linear(device const float *w [[buffer(0)]],device const float *sc [[buffer(1)]],device const float *off [[buffer(2)]],device const uchar *bs [[buffer(3)]],device const float *x [[buffer(4)]],device const float *bias [[buffer(5)]],device float *y [[buffer(6)]],constant uint *p [[buffer(7)]],uint tid [[thread_index_in_threadgroup]],uint row [[threadgroup_position_in_grid]]) {
 uint cols=p[1],lanes=cols<=64?4:32;float sum=0;
 if(tid<lanes)for(uint k=tid*4;k<cols;k+=lanes*4)for(uint j=0;j<4 && k+j<cols;j++)sum+=x[ulong(row)*cols+k+j]*w[k+j];
 if(lanes==32){sum+=simd_shuffle_down(sum,ushort(16));sum+=simd_shuffle_down(sum,ushort(8));sum+=simd_shuffle_down(sum,ushort(4));}
 sum+=simd_shuffle_down(sum,ushort(2));sum+=simd_shuffle_down(sum,ushort(1));if(tid==0)y[row]=sum+(p[5]?bias[0]:0.0f);
}
kernel void audio_source_normal(device const float *x [[buffer(0)]],device const float *unused [[buffer(1)]],device float *y [[buffer(2)]],constant uint *p [[buffer(3)]],uint gid [[thread_position_in_grid]]) {if(gid<p[0])y[gid]=music_erfinv(x[gid])*1.4142135623730951f;}
