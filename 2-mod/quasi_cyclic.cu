#include "quasi_cyclic.h"
#include <cmath>
#include "gpu_vector.h"
#include "gpu_ops.h"

#define DENSITY 4096

QuasiCyclic::QuasiCyclic(uint64_t in, uint64_t out) : mIn(in), mOut(out) {
  if (mIn == 0 || mOut == 0) return;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, 0);
  nBlocks = (mOut + rows - 1) / rows;
  n2Blocks = ((mIn - mOut) + rows - 1) / rows;
  n64 = nBlocks * 2;

  cufftCreate(&aPlan);
  cufftCreate(&bPlan);
  cufftCreate(&cPlan);
  cufftPlan1d(&aPlan, n64, CUFFT_C2C, 1);
  cufftPlan1d(&bPlan, n64, CUFFT_C2C, rows);
  cufftPlan1d(&cPlan, n64, CUFFT_C2C, rows);
}

QuasiCyclic::~QuasiCyclic() {
  if (mIn == 0 || mOut == 0) return;
  curandDestroyGenerator(prng);
}

__global__
void load_column(OTblock *o, OTblock *i, uint64_t c, uint64_t numCols) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  o[tid] = i[0 * numCols + c];
}

__global__
void xor_column(OTblock *out, OTblock *in, uint64_t vecStart) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 4; i++) {
    out[tid + vecStart].data[i] ^= in[tid].data[i];
  }
}

void QuasiCyclic::encode(GPUvector<OTblock> &vector) {
  GPUmatrix<OTblock> XT(mOut, 1); // XT = mOut x 1
  XT.load((uint8_t*) vector.data());
  XT.bit_transpose(); // XT = 128 x n2blocks

  GPUmatrix<OTblock> cModP1(rows, 2 * nBlocks);

  GPUvector<OTblock> temp128(n64);
  uint64_t *a64 = (uint64_t*) temp128.data();
  cufftReal *a64_poly;
  cufftComplex *a64_fft;
  curandGenerateLongLong(prng, (unsigned long long*) a64, n64);
  cudaMalloc(&a64_poly, n64 * sizeof(cufftReal));
  cudaMalloc(&a64_fft, n64 * sizeof(cufftComplex));
  int_to_float<<<n64 / 1024, 1024>>>(a64_poly, a64);
  cudaDeviceSynchronize();
  cufftExecR2C(aPlan, a64_poly, a64_fft);

  uint64_t *b64 = (uint64_t*) XT.data();
  cufftReal *b64_poly;
  cufftComplex *b64_fft;
  cudaMalloc(&b64_poly, n64 * sizeof(cufftReal));
  cudaMalloc(&b64_fft, n64 * sizeof(cufftComplex));
  int_to_float<<<n64 / 1024, 1024>>>(b64_poly, a64);
  cudaDeviceSynchronize();
  cufftExecR2C(bPlan, b64_poly, b64_fft);

  cufftComplex *c64_fft;
  cufftReal *c64_poly;
  cudaMalloc(&c64_poly, n64 * sizeof(cufftReal));
  cudaMalloc(&c64_fft, n64 * sizeof(cufftComplex));
  complex_dot_product<<<n64 / 1024, 1024>>>(c64_fft, a64_fft, b64_fft);
  cudaDeviceSynchronize();
  cufftExecC2R(cPlan, c64_fft, c64_poly);
  float_to_int<<<n64 / 1024, 1024>>>((uint64_t*) cModP1.data(), c64_poly);

  cModP1.modp(mOut);

  GPUvector<OTblock> tpBuffer(rows);
  uint64_t numBlocks = (mOut + rows - 1) / rows;
  for (uint64_t i = 0; i < numBlocks; i++) {
    uint64_t j = i * rows;
    uint64_t min = std::min<uint64_t>(rows, mOut - j);

    load_column<<<1, rows>>>(tpBuffer.data(), cModP1.data(), i, cModP1.cols());
    cudaDeviceSynchronize();

    xor_column<<<1, rows>>>(vector.data(), tpBuffer.data(), j);
    cudaDeviceSynchronize();
  }
}
