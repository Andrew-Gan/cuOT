#include "quasi_cyclic.h"
#include <cmath>
#include "gpu_tests.h"
#include "gpu_matrix.h"
#include "gpu_ops.h"
#include "pprf.h"
#include "logger.h"
#include "silent_ot.h"

#ifndef USE_CUFFT_FOR_POLYMUL
#include "exampleFFT.h"
#endif

#ifndef USE_IMPROVED_COMPLEX_PROD
#include <cublas_v2.h>
#endif

#define FFT_BATCHSIZE 8

__global__ void bit_to_float(uint8_t *bitPoly, cufftReal *fftReal, uint64_t inBitWidth, uint64_t outFloatWidth) {
  uint64_t row = blockIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t tmp = bitPoly[row * inBitWidth / 8 + col];
  uint64_t offset = row * outFloatWidth + 8 * col;
  for (int j = 0; j < 8; j++) {
    fftReal[offset + j] = (cufftReal)(tmp & 1);
    tmp >>= 1;
  }
}

#ifdef USE_TYPE_CONVERT_AND_MOD
__global__ void float_to_bit_and_modp(cufftReal *fftReal, uint8_t *bitPoly, uint64_t n) {
  uint64_t row = blockIdx.y;
  uint64_t mOut = 8 * gridDim.x * blockDim.x;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t offset = row * n + 8 * col;
  uint8_t res = 0;
  for (int i = 0; i < n / mOut; i++) {
    for (int j = 0; j < 8; j++)
    {
      // divide float by FFT size to obtain true result
      if ((uint64_t)fftReal[offset + (i * mOut) + j] & n)
        res ^= 1UL << j;
    }
  }
  bitPoly[row * (mOut / 8) + col] = res;
}
#else
__global__ void float_to_bit(cufftReal *fftReal, uint8_t *bitPoly, uint64_t n) {
  uint64_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t offset = 8 * byte_idx;
  uint8_t res = 0;
  for (int i = 0; i < 8; i++) {
    // divide float by FFT size to obtain true result
    if (fftReal[offset + i] / n)
      res |= 1UL << i;
  }
  bitPoly[byte_idx] = res;
}
#endif // USE_TYPE_CONVERT_AND_MOD

QuasiCyclic::QuasiCyclic(Role role, uint64_t in, uint64_t out, int rows) : mRole(role), mIn(in), mOut(out), mRows(rows) {

  a.resize({mOut / BLOCK_BITS});
  make_block<<<a.size() / 1024, 1024>>>(a.data());
  blk key;
  for (int i = 0; i < 4; i++)
    key.data[i] = rand();
  Aes aes(&key);
  aes.encrypt(a);

#ifdef USE_CUFFT_FOR_POLYMUL
  cudaMalloc(&a_poly, mIn * sizeof(cufftReal));
  cudaMemset(a_poly+mOut, 0, mOut * sizeof(cufftReal));
  bit_to_float<<<mOut / 8 / 1024, 1024>>>((uint8_t *)a.data(), a_poly, mOut, mIn);

  cufftHandle aPlan;
  cufftCreate(&aPlan);
  cufftPlan1d(&aPlan, mIn, CUFFT_R2C, 1);
  cudaMalloc(&a_fft, (mIn / 2 + 1) * sizeof(cufftComplex));
  cufftExecR2C(aPlan, a_poly, a_fft);
  cudaFree(a_poly);
  cufftDestroy(aPlan);

  cufftCreate(&bPlan);
  cufftCreate(&cPlan);
  cufftSetAutoAllocation(bPlan, 0);
  cufftSetAutoAllocation(cPlan, 0);
  size_t bSize, cSize;
  cufftMakePlan1d(bPlan, mIn, CUFFT_R2C, FFT_BATCHSIZE, &bSize);
  cufftMakePlan1d(cPlan, mIn, CUFFT_C2R, FFT_BATCHSIZE, &cSize);
  cudaMalloc(&workArea, std::max(bSize, cSize));
  cufftSetWorkArea(bPlan, workArea);
  cufftSetWorkArea(cPlan, workArea);

  cudaMalloc(&b_poly, FFT_BATCHSIZE * mIn * sizeof(cufftReal));
  cudaMemset2D(b_poly+mOut, mIn * sizeof(cufftReal), 0, mOut * sizeof(cufftReal), FFT_BATCHSIZE);
  cudaMalloc(&b_fft, FFT_BATCHSIZE * (mIn / 2 + 1) * sizeof(cufftComplex));
  cudaMalloc(&c_poly, FFT_BATCHSIZE * mIn * sizeof(cufftReal));
#else
  cudaMalloc(&workArea, mIn * sizeof(float2));
  cudaMalloc(&a_fft, mIn * sizeof(*a_fft));
  cudaMalloc(&b_fft, mIn * sizeof(*b_fft));
  cudaMemset(a_fft+mOut, 0, mOut * sizeof(*a_fft));
  cudaMemset(b_fft+mOut, 0, mOut * sizeof(*b_fft));
  preprocess<<<mOut / 64 / 1024, 1024>>>((uint64_t *)a.data(), a_fft);
  fft(a_fft, (float2*)workArea, mIn / 64, false, mOut / 64, 1024); // FFT(a)
#endif // USE_CUFFT_FOR_POLYMUL
}

QuasiCyclic::~QuasiCyclic() {
  cudaFree(workArea);
  cudaFree(a_fft);
  cudaFree(b_fft);
#ifdef USE_CUFFT_FOR_POLYMUL
  cufftDestroy(bPlan);
  cufftDestroy(cPlan);
  cudaFree(b_poly);
  cudaFree(c_poly);
#endif
}

__global__
void complex_product(float2 *in, float2 *io, uint64_t len) {
  uint64_t c = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t r = blockIdx.y;
  if (c >= len)
    return;
  float2 a = in[c], b = io[r * len + c];
  b.x = a.x * b.x - a.y * b.y;
  b.y = a.x * b.y + a.y * b.x;
  io[r * len + c] = b;
}

void QuasiCyclic::encode_dense(Mat &b64) {
#ifdef USE_CUFFT_FOR_POLYMUL
  uint64_t thread = mIn / 2 + 1;
  uint64_t blockCplxProd = std::min(thread, 1024UL);
  dim3 gridCplxProd((thread + blockCplxProd - 1) / blockCplxProd, FFT_BATCHSIZE);

  thread = mOut / 8;
  uint64_t blockConvert = std::min(thread, 1024UL);
  dim3 gridConvert((thread + blockConvert - 1) / blockConvert, FFT_BATCHSIZE);

  for (uint64_t r = 0; r < mRows; r += FFT_BATCHSIZE) {
    bit_to_float<<<gridConvert, blockConvert>>>((uint8_t *)b64.data({r, 0}), b_poly, mOut, mIn);
    cufftExecR2C(bPlan, b_poly, b_fft);
#ifdef USE_IMPROVED_COMPLEX_PROD
    complex_product<<<gridCplxProd, blockCplxProd>>>(a_fft, b_fft, mIn / 2 + 1);
#else
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasCdgmm(handle, CUBLAS_SIDE_RIGHT, 1, mIn / 2 + 1, a_fft, 1, b_fft, 1, b_fft, 1);
    cublasDestroy(handle);
#endif
    cufftExecC2R(cPlan, b_fft, c_poly);

#ifdef USE_TYPE_CONVERT_AND_MOD
    float_to_bit_and_modp<<<gridConvert, blockConvert>>>(c_poly, (uint8_t *)b64.data({r, 0}), mIn);
#else
    Mat cModP1({FFT_BATCHSIZE, mIn / (8 * sizeof(blk))});
    float_to_bit<<<mIn * FFT_BATCHSIZE / 8 / 1024, 1024>>>(c_poly, (uint8_t *)cModP1.data(), mIn);
    cModP1.modp(mOut / BLOCK_BITS);
#endif // USE_TYPE_CONVERT_AND_MOD
  }
#else
  fft(b_fft, (float2*)workArea, mIn, false, mOut, 1024); // FFT(b)
  complex_product<<<mIn / 1024, 1024>>>(a_fft, b_fft, mIn);
  fft(b_fft, (float2*)workArea, mIn / 64, true, mOut, 1024); // IFFT(A * B)
#endif // USE_CUFFT_FOR_POLYMUL
}

__global__ void choice_vec_prod(uint64_t *mat, uint64_t *vec, uint64_t weight, uint64_t *out) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t mOut = 64 * gridDim.x * blockDim.x;
  uint64_t src = mat[i];
  // each thread divides up src vector
  for (int w = 0; w < weight && vec[w] < mOut; w++) {
    uint64_t new_pow = 64 * i + vec[w];
    out[new_pow / 64] ^= src << (vec[w] % 64);
    out[new_pow / 64 + 1] ^= src >> (64 - vec[w] % 64);
  }
}

void QuasiCyclic::encode_sparse(Mat &out, uint64_t *sparsePos, int weight) {
  out.resize({mIn / BLOCK_BITS});
  out.clear();

  uint64_t nThread = mOut / 64;
  uint64_t block = std::min(1024UL, nThread);
  uint64_t grid = (nThread + block - 1) / block;
  uint64_t *a64 = (uint64_t *)a.data();
  uint64_t *out64 = (uint64_t *)out.data();
  choice_vec_prod<<<grid, block>>>(a64, sparsePos, weight, out64);
  out.modp(mOut / BLOCK_BITS);
  out.resize({mOut / BLOCK_BITS});
}
