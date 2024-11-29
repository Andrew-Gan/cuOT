#ifndef __EXAMPLE_FFT_H__
#define __EXAMPLE_FFT_H__

#include <bits/stdc++.h>

static __device__ __host__ inline float2 Add(float2 A, float2 B) {
    float2 C;
    C.x = A.x + B.x;
    C.y = A.y + B.y;
    return C;
}

static __device__ __host__ inline float2 Inverse(float2 A) {
    float2 C;
    C.x = -A.x;
    C.y = -A.y;
    return C;
}

static __device__ __host__ inline float2 Multiply(float2 A, float2 B) {
    float2 C;
    C.x = A.x * B.x - A.y * B.y;
    C.y = A.y * B.x + A.x * B.y;
    return C;
}

__global__ void inplace_divide_invert(float2 *A, int n, int threads) {
    int i = blockIdx.x * threads + threadIdx.x;
    if (i < n) {
        A[i].x /= n;
        A[i].y /= n;
    }
}

__global__ void bitrev_reorder(float2 *__restrict__ r, float2 *__restrict__ d, int s, size_t nthr, int n) {
    int id = blockIdx.x * nthr + threadIdx.x;
    if (id < n and __brev(id) >> (32 - s) < n)
        r[__brev(id) >> (32 - s)] = d[id];
}

__device__ void inplace_fft_inner(float2 *__restrict__ A, int i, int j, int len, int n, bool invert) {
    if (i + j + len / 2 < n and j < len / 2) {
        float2 u, v;

        float angle = (2 * M_PI * j) / (len * (invert ? -1.0 : 1.0));
        v.x = cos(angle);
        v.y = sin(angle);

        u = A[i + j];
        v = Multiply(A[i + j + len / 2], v);
        A[i + j] = Add(u, v);
        A[i + j + len / 2] = Add(u, Inverse(v));
    }
}

__global__ void inplace_fft(float2 *__restrict__ A, int i, int len, int n, int threads, bool invert) {
    int j = blockIdx.x * threads + threadIdx.x;
    inplace_fft_inner(A, i, j, len, n, invert);
}

__global__ void inplace_fft_outer(float2 *__restrict__ A, int len, int n, int threads, bool invert) {
    int i = (blockIdx.x * threads + threadIdx.x);
    for (int j = 0; j < len / 2; j++) {
        inplace_fft_inner(A, i, j, len, n, invert);
    }
}

void fft(float2 *dn, float2 *A, int n, bool invert, int balance = 10, int threads = 32) {
    // Bit reversal reordering
    int s = log2(n);

    bitrev_reorder<<<ceil(float(n) / threads), threads>>>(A, dn, s, threads, n);
    // Iterative FFT with loop parallelism balancing
    for (int len = 2; len <= n; len <<= 1) {
        if (n / len > balance) {
            inplace_fft_outer<<<ceil((float)n / threads), threads>>>(A, len, n, threads, invert);
        }
        else {
            for (int i = 0; i < n; i += len) {
                float repeats = len / 2;
                inplace_fft<<<ceil(repeats / threads), threads>>>(A, i, len, n, threads, invert);
            }
        }
    }
    
    if (invert)
        inplace_divide_invert<<<ceil(n * 1.00 / threads), threads>>>(A, n, threads);

    cudaMemcpy(dn, A, n * sizeof(float2), cudaMemcpyDeviceToDevice);
}

__global__
void preprocess(uint64_t* b, float2 *f) {
    uint64_t byte = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t bit = 8 * byte;
    uint64_t src = b[byte];
    for (int b = bit; b < bit + 8; b++) {
        f[b].x = (__half)(src & 0b1);
        f[b].y = 0;
        src >>= 1;
    }
}

__global__
void postprocess(float2 *f, uint64_t* b) {
    uint64_t byte = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t bit = 64 * byte;
    uint64_t n = gridDim.x * blockDim.x;
    uint64_t res = 0;
    for (int b = bit; b < bit + 64; b++) {
        res <<= 1;
        res |= (int)(f[b].x + f[b + n].x) % 2;
    }
    b[byte] = res;
}

#endif
