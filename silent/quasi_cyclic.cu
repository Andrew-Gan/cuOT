#include "lpn.h"
#include <cmath>
#include "gpu_tests.h"
#include "gpu_vector.h"
#include "gpu_ops.h"
#include "logger.h"

#define FFT_BATCHSIZE 16

__global__
void bit_to_float(uint64_t *bitPoly, cufftReal *fftReal, uint64_t inBitWidth, uint64_t outFloatWidth) {
  uint64_t row = blockIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t tmp = bitPoly[row * (inBitWidth / 64) + col];
  uint64_t offset = row * outFloatWidth + 64 * col;
  for (int j = 0; j < 64; j++) {
    fftReal[offset++] = (cufftReal)(tmp & 1);
    tmp >>= 1;
  }
}

__global__
void complex_dot_product(cufftComplex *in, cufftComplex *io, uint64_t len) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t r = blockIdx.y;
  if (x >= len) return;

  cufftComplex a = in[x], b = io[r*len+x], c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  io[r*len+x] = c;
}

__global__
void float_to_bit(cufftReal *fftReal, uint64_t *bitPoly, uint64_t mIn, uint64_t scaleLog) {
  uint64_t row = blockIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t res = 0;
  uint64_t offset = row * mIn + 64 * col;
  uint64_t pow2 = 1 << scaleLog;
  for (int j = 0; j < 64; j++) {
    if (lround(fftReal[offset++]) & pow2) {
      res |= 1UL << j;
    }
  }
  bitPoly[row * (mIn / 64) + col] = res;
}

QuasiCyclic::QuasiCyclic(Role role, uint64_t in, uint64_t out, int rows) :
  mRole(role), mIn(in), mOut(out), mRows(rows) {

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

  cufftReal *a64_poly;
  cudaMalloc(&a64_poly, mIn * sizeof(cufftReal));
  cudaMemset(a64_poly, 0, mIn * sizeof(cufftReal));
  cudaMalloc(&a64_fft, (mIn / 2 + 1) * sizeof(cufftComplex));
  cudaMalloc(&b64_poly, FFT_BATCHSIZE * mIn * sizeof(cufftReal));
  cudaMemset(b64_poly, 0, FFT_BATCHSIZE * mIn * sizeof(cufftReal));
  cudaMalloc(&b64_fft, FFT_BATCHSIZE * (mIn / 2 + 1) * sizeof(cufftComplex));
  cudaMalloc(&c64_poly, FFT_BATCHSIZE * mIn * sizeof(cufftReal));

  a64.resize(mOut / BLOCK_BITS);
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 1234);
  curandGenerate(prng, (uint32_t*)a64.data(), 4 * a64.size());
  curandDestroyGenerator(prng);
  uint64_t thread = mOut / 64;
  uint64_t block = std::min(thread, 1024UL);
  uint64_t grid = (thread + block - 1) / block;
  bit_to_float<<<grid, block>>>((uint64_t*)a64.data(), a64_poly, mOut, mIn);

  cufftHandle aPlan;
  cufftCreate(&aPlan);
  cufftPlan1d(&aPlan, mIn, CUFFT_R2C, 1);
  cufftExecR2C(aPlan, a64_poly, a64_fft);
  cudaFree(a64_poly);
  cufftDestroy(aPlan);

  cModP1.resize({mRows, mIn / BLOCK_BITS});
  cModP1.clear();

  uint64_t tmp = mIn;
  while(tmp != 0) {
    tmp >>= 1;
    fftsizeLog++;
  }
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

void QuasiCyclic::encode_dense(Mat &b64) {
  Log::mem(mRole, LPN);
  // bitpoly to fft
  uint64_t thread1 = mOut / 64;
  uint64_t block1 = std::min(thread1, 1024UL);
  dim3 grid1((thread1 + block1 - 1) / block1, FFT_BATCHSIZE);
  // complex dot product and divider
  uint64_t thread2 = mIn / 2 + 1;
  uint64_t block2 = std::min(thread2, 1024UL);
  dim3 grid2((thread2 + block2 - 1) / block2, FFT_BATCHSIZE);
  // fft to bitpoly
  uint64_t thread3 = mIn / 64;
  uint64_t block3 = std::min(thread3, 1024UL);
  dim3 grid3((thread3 + block3 - 1) / block3, FFT_BATCHSIZE);

  for (uint64_t r = 0; r < mRows; r += FFT_BATCHSIZE) {
    bit_to_float<<<grid1, block1>>>((uint64_t*) b64.data({r, 0}), b64_poly, mOut, mIn);
    cufftExecR2C(bPlan, b64_poly, b64_fft);
    complex_dot_product<<<grid2, block2>>>(a64_fft, b64_fft, thread2);
    cufftExecC2R(cPlan, b64_fft, c64_poly);
    float_to_bit<<<grid3, block3>>>(c64_poly, (uint64_t*)cModP1.data({r, 0}), mIn, fftsizeLog);
  }

  cModP1.modp(mOut / BLOCK_BITS);
  cModP1.resize({cModP1.dim(0), mOut / BLOCK_BITS});
  b64 = cModP1;
  Log::mem(mRole, LPN);
}

__global__
void cyclic_mat_vec_prod(uint64_t *mat, uint64_t *vec, uint64_t weight, uint64_t *out) {
  uint64_t r64 = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t num64 = gridDim.x * blockDim.x;
  uint64_t alignment;
  uint64_t op;

  for (int i = 0; i < weight; i++) {
    alignment = vec[i] % 64;
    op = 0;
    if (r64 < num64 - 1) op |= mat[r64] << alignment;
    if (r64 > 0) op |= mat[r64-1] >> (64-alignment);
    out[vec[i] / 64 + r64] |= op;
  }
}

void QuasiCyclic::encode_sparse(Vec &out, uint64_t *sparsePos, int weight) {
  Log::mem(mRole, LPN);
  out.resize(mIn / BLOCK_BITS);
  uint64_t nThread = mOut / 64;
  uint64_t block = std::min(1024UL, nThread);
  uint64_t grid = (nThread + block - 1) / block;
  cyclic_mat_vec_prod<<<grid, block>>>((uint64_t*)a64.data(), sparsePos, weight, (uint64_t*)out.data());
  out.modp(mOut / BLOCK_BITS);
  out.resize(mOut / BLOCK_BITS);
  Log::mem(mRole, LPN);
}
