#include "quasi_cyclic.h"
#include <cmath>
#include "gpu_tests.h"
#include "gpu_matrix.h"
#include "gpu_ops.h"
#include "pprf.h"
#include "logger.h"

#define FFT_BATCHSIZE 16

__global__
void bit_to_float(uint8_t *bitPoly, cufftReal *fftReal, uint64_t inBitWidth, uint64_t outFloatWidth) {
  uint64_t row = blockIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t tmp = bitPoly[row * inBitWidth / 8 + col];
  uint64_t offset = row * outFloatWidth + 8 * col;
  for (int j = 0; j < 8; j++) {
    fftReal[offset+j] = (cufftReal)(tmp & 1);
    tmp >>= 1;
  }
}

__global__
void complex_dot_product(cufftComplex *in, cufftComplex *io, uint64_t len) {
  uint64_t c = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t r = blockIdx.y;
  if (c >= len) return;

  cufftComplex a = in[c], b = io[r*len+c];
  io[r*len+c].x = a.x * b.x - a.y * b.y;
  io[r*len+c].y = a.x * b.y + a.y * b.x;
}

__global__
void float_to_bit_and_modp(cufftReal *fftReal, uint8_t *bitPoly, uint64_t mIn) {
  uint64_t row = blockIdx.y;
  uint64_t mOut = 8 * gridDim.x * blockDim.x;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint8_t res = 0;
  uint64_t offset = row * mIn + 8 * col;
  for (int i = 0; i < mIn / mOut; i++) {
    for (int j = 0; j < 8; j++) {
      if ((uint64_t)fftReal[offset+(i*mOut)+j] & mIn) {
        res ^= 1UL << j;
      }
    }
  }
  bitPoly[row * (mOut / 64) + col] = res;
}

QuasiCyclic::QuasiCyclic(Role role, uint64_t in, uint64_t out, int rows) :
  mRole(role), mIn(in), mOut(out), mRows(rows) {

  cufftReal *a64_poly;
  cudaMalloc(&a64_poly, mIn * sizeof(cufftReal));
  cudaMemset(a64_poly, 0, mIn * sizeof(cufftReal));
  cudaMalloc(&a64_fft, (mIn / 2 + 1) * sizeof(cufftComplex));
  a64.resize({mOut / BLOCK_BITS});
  
  blk key;
  for (int i = 0; i < 4; i++) {
    key.data[i] = rand();
  }
  make_block<<<a64.size() / 1024, 1024>>>(a64.data());
  Aes aes(&key);
  aes.encrypt(a64);

  uint64_t thread = mOut / 8;
  uint64_t block = std::min(thread, 1024UL);
  uint64_t grid = (thread + block - 1) / block;
  bit_to_float<<<grid, block>>>((uint8_t*)a64.data(), a64_poly, mOut, mIn);
  cufftHandle aPlan;
  cufftCreate(&aPlan);
  cufftPlan1d(&aPlan, mIn, CUFFT_R2C, 1);
  cufftExecR2C(aPlan, a64_poly, a64_fft);
  cudaFree(a64_poly);
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
  cudaMalloc(&b64_poly, FFT_BATCHSIZE * mIn * sizeof(cufftReal));
  cudaMemset(b64_poly, 0, FFT_BATCHSIZE * mIn * sizeof(cufftReal));
  cudaMalloc(&b64_fft, FFT_BATCHSIZE * (mIn / 2 + 1) * sizeof(cufftComplex));
  cudaMalloc(&c64_poly, FFT_BATCHSIZE * mIn * sizeof(cufftReal));

  // bitpoly to fft
  thread = mOut / 8;
  blockFFT[0] = std::min(thread, 1024UL);
  gridFFT[0] = dim3((thread + blockFFT[0] - 1) / blockFFT[0], FFT_BATCHSIZE);
  // complex dot product and divider
  thread = mIn / 2 + 1;
  blockFFT[1] = std::min(thread, 1024UL);
  gridFFT[1] = dim3((thread + blockFFT[1] - 1) / blockFFT[1], FFT_BATCHSIZE);
  // fft to bitpoly
  thread = mOut / 8;
  blockFFT[2] = std::min(thread, 1024UL);
  gridFFT[2] = dim3((thread + blockFFT[2] - 1) / blockFFT[2], FFT_BATCHSIZE);
}

QuasiCyclic::~QuasiCyclic() {
  cufftDestroy(bPlan);
  cufftDestroy(cPlan);
  cudaFree(workArea);
  cudaFree(a64_fft);
  cudaFree(b64_poly);
  cudaFree(b64_fft);
  cudaFree(c64_poly);
}

void QuasiCyclic::encode_dense(Span &b64) {
  Log::mem(mRole, LPN);
  for (uint64_t r = 0; r < mRows; r += FFT_BATCHSIZE) {
    bit_to_float<<<gridFFT[0], blockFFT[0]>>>((uint8_t*)b64.data({r, 0}), b64_poly, mIn, mIn);
    cufftExecR2C(bPlan, b64_poly, b64_fft);
    complex_dot_product<<<gridFFT[1], blockFFT[1]>>>(a64_fft, b64_fft, mIn / 2 + 1);
    cufftExecC2R(cPlan, b64_fft, c64_poly);
    float_to_bit_and_modp<<<gridFFT[2], blockFFT[2]>>>(c64_poly, (uint8_t*)b64.data({r, 0}), mIn);
  }
  Log::mem(mRole, LPN);
}

__global__
void cyclic_mat_vec_prod(uint64_t *mat, uint64_t *vec, uint64_t weight, uint64_t *out, uint64_t mOut, uint64_t n) {
  uint64_t r64 = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t alignment;
  uint64_t op;

  if (r64 >= n) return;

  for (int i = 0; i < weight; i++) {
    if (vec[i] > mOut) continue;
    alignment = vec[i] % 64;
    op = 0;
    if (r64 < n - 1) op |= mat[r64] << alignment;
    if (r64 > 0) op |= mat[r64-1] >> (64-alignment);
    out[vec[i] / 64 + r64] ^= op;
  }
}

void QuasiCyclic::encode_sparse(Mat &out, uint64_t *sparsePos, int weight) {
  Log::mem(mRole, LPN);
  out.resize({mIn / BLOCK_BITS});
  out.clear();
  uint64_t nThread = mOut / 64 + 1;
  uint64_t block = std::min(1024UL, nThread);
  uint64_t grid = (nThread + block - 1) / block;
  cyclic_mat_vec_prod<<<grid, block>>>(
    (uint64_t*)a64.data(), sparsePos, weight, (uint64_t*)out.data(), mOut, nThread
  );
  out.modp(mOut / BLOCK_BITS);
  out.resize({mOut / BLOCK_BITS});
  Log::mem(mRole, LPN);
}
