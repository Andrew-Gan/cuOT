#include "quasi_cyclic.h"
#include <cmath>
#include "gpu_vector.h"
#include "gpu_ops.h"

#define DENSITY 4096

QuasiCyclic::QuasiCyclic(Role role, uint64_t in, uint64_t out) : mRole(role), mIn(in), mOut(out) {

  if (mIn == 0 || mOut == 0) return;
  Log::start(mRole, CompressInit);
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, 0);
  nBlocks = (mOut + rows - 1) / rows;
  n2Blocks = ((mIn - mOut) + rows - 1) / rows;
  n64 = nBlocks * 2;

  cufftCreate(&aPlan);
  cufftCreate(&bPlan);
  cufftCreate(&cPlan);
  cufftPlan1d(&aPlan, n64, CUFFT_R2C, 1);
  cufftPlan1d(&bPlan, n64, CUFFT_R2C, rows);
  cufftPlan1d(&cPlan, n64, CUFFT_C2R, rows);
  Log::end(mRole, CompressInit);
}

QuasiCyclic::~QuasiCyclic() {
  if (mIn == 0 || mOut == 0) return;
  curandDestroyGenerator(prng);
  cufftDestroy(aPlan);
  cufftDestroy(bPlan);
  cufftDestroy(cPlan);
}

void QuasiCyclic::encode(GPUvector<OTblock> &vector) {
  // if (mRole == Sender) std::cout << vector << std::endl;
  Log::start(mRole, CompressTP);
  GPUmatrix<OTblock> XT(mOut, 1); // XT = mOut x 1
  XT.load((uint8_t*) vector.data());
  XT.bit_transpose(); // XT = rows x n2blocks
  Log::end(mRole, CompressTP);

  Log::start(mRole, CompressFFT);
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
  cudaMalloc(&b64_poly, rows * n64 * sizeof(cufftReal));
  cudaMalloc(&b64_fft, rows * n64 * sizeof(cufftComplex));
  int_to_float<<<rows * n64 / 1024, 1024>>>(b64_poly, b64);
  cudaDeviceSynchronize();
  cufftExecR2C(bPlan, b64_poly, b64_fft);
  Log::end(mRole, CompressFFT);

  Log::start(mRole, CompressMult);
  cufftComplex *c64_fft;
  cufftReal *c64_poly;
  cudaMalloc(&c64_poly, rows * n64 * sizeof(cufftReal));
  cudaMalloc(&c64_fft, rows * n64 * sizeof(cufftComplex));
  dim3 blocks(n64 / 1024, rows);
  complex_dot_product<<<blocks, 1024>>>(c64_fft, a64_fft, b64_fft);
  cudaDeviceSynchronize();
  Log::end(mRole, CompressMult);

  Log::start(mRole, CompressIFFT);
  cufftExecC2R(cPlan, c64_fft, c64_poly);
  Log::end(mRole, CompressIFFT);

  Log::start(mRole, CompressTP);
  GPUmatrix<OTblock> cModP1(rows, 2 * nBlocks); // hold unmodded coeffs
  float_to_int<<<rows * n64 / 1024, 1024>>>((uint64_t*) cModP1.data(), c64_poly);

  cModP1.modp(nBlocks); // cModP1 = rows x nBlocks
  cModP1.bit_transpose(); // cModP1 = mOut x 1

  xor_gpu<<<16 * mOut / 1024, 1024>>>((uint8_t*) vector.data(), (uint8_t*) cModP1.data(), 16 * mOut);
  cudaDeviceSynchronize();
  Log::end(mRole, CompressTP);
}
