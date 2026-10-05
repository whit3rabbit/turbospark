// Softmax reduction adapted from MLX 0.32.3 mlx/backend/metal/kernels/softmax.h.
// Copyright (c) 2023-2024 Apple Inc.
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

// Composed after music3.metal and compiled with precise math. All buffers store
// logical F32/F16/BF16 values as floats; music_round applies the native casts.
// p: batch, queries, keys, heads, kv_heads, dim, time_major, causal, offset,
// threads, dtype, scale_bits. Q/output are BQHD; K/V are BKHD or KBHD.
kernel void music3_attention_fallback_qk(
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
                ? music_round(q[((ulong(b)*p[1]+query)*p[3]+h)*dim+d]
                    * as_type<float>(p[11]),p[10]) : 0.0f;
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
                : music_round(result[idx],p[10]);
        }
    }
}

// Four reads per thread and the two SIMD reductions preserve MLX's FP32
// normalizer before the reciprocal multiply and final logical dtype cast.
kernel void music3_attention_fallback_softmax(
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
            if(key<keys) probabilities[base+key]=music_round(values[i]*normalizer,p[10]);
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

kernel void music3_attention_fallback_pv(
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
            =music_round(result[idx],p[10]);
    }
}
