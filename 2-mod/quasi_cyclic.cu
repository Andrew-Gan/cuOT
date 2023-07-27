#include "quasi_cyclic.h"
#include <cmath>
#include "gpu_vector.h"
#include "gpu_ops.h"

#define DENSITY 4096

QuasiCyclic::QuasiCyclic(Role role, uint64_t in, uint64_t out) : mRole(role), mIn(in), mOut(out) {
  if (mIn == 0 || mOut == 0) return;
  Log::start(mRole, CompressInit);
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 50);
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

  Log::start(mRole, CompressFFT);
  GPUvector<OTblock> a64(n64);
  cufftReal *a64_poly;
  curandGenerate(prng, (uint32_t*) a64.data(), 2 * n64);

  cudaMalloc(&a64_poly, n64 * sizeof(cufftReal));
  cudaMalloc(&a64_fft, n64 * sizeof(cufftComplex));

  uint64_t blk = std::min(n64, 1024lu);
  uint64_t grid = n64 < 1024 ? 1 : n64 / 1024;
  cast<uint64_t, cufftReal><<<grid, blk>>>((uint64_t*) a64.data(), a64_poly);
  cudaDeviceSynchronize();

  cufftExecR2C(aPlan, a64_poly, a64_fft);
  cudaFree(a64_poly);
  Log::end(mRole, CompressFFT);
}

QuasiCyclic::~QuasiCyclic() {
  if (mIn == 0 || mOut == 0) return;
  curandDestroyGenerator(prng);
  cufftDestroy(aPlan);
  cufftDestroy(bPlan);
  cufftDestroy(cPlan);
  cudaFree(a64_fft);
}

void QuasiCyclic::encode(GPUvector<OTblock> &vector) {
  Log::start(mRole, CompressTP);
  GPUmatrix<OTblock> XT(mOut, 1); // XT = mOut x 1
  XT.load((uint8_t*) vector.data());
  XT.bit_transpose(); // XT = rows x n2blocks
  Log::end(mRole, CompressTP);

  Log::start(mRole, CompressFFT);
  uint64_t *b64 = (uint64_t*) XT.data();
  cufftReal *b64_poly;
  cufftComplex *b64_fft;
  cudaMalloc(&b64_poly, rows * n64 * sizeof(cufftReal));
  cudaMalloc(&b64_fft, rows * n64 * sizeof(cufftComplex));

  uint64_t blk = std::min(rows * n64, 1024lu);
  uint64_t grid = rows * n64 < 1024 ? 1 : rows * n64 / 1024;
  cast<uint64_t, cufftReal><<<grid, blk>>>(b64, b64_poly);
  cudaDeviceSynchronize();

  cufftExecR2C(bPlan, b64_poly, b64_fft);
  cudaFree(b64_poly);
  Log::end(mRole, CompressFFT);

  Log::start(mRole, CompressMult);
  cufftComplex *c64_fft;
  cufftReal *c64_poly;
  cudaMalloc(&c64_poly, rows * n64 * sizeof(*c64_poly));
  cudaMalloc(&c64_fft, rows * n64 * sizeof(*c64_fft));

  blk = std::min(n64, 1024lu);
  dim3 blocks(n64 < 1024 ? 1 : n64 / 1024, rows);
  complex_dot_product<<<blocks, blk>>>(c64_fft, a64_fft, b64_fft);
  cudaDeviceSynchronize();
  cudaFree(b64_fft);
  Log::end(mRole, CompressMult);

  Log::start(mRole, CompressIFFT);
  cufftExecC2R(cPlan, c64_fft, c64_poly);
  cudaFree(c64_fft);
  Log::end(mRole, CompressIFFT);

  printf("c64 poly\n");
  print_gpu<<<1, 1>>>(c64_poly, 64, 16);
  cudaDeviceSynchronize();

  Log::start(mRole, CompressTP);
  GPUmatrix<OTblock> cModP1(rows, 2 * nBlocks); // hold unmodded coeffs
  cast<cufftReal, uint64_t><<<rows * n64 / 1024, 1024>>>(c64_poly, (uint64_t*) cModP1.data());
  cudaDeviceSynchronize();
  cudaFree(c64_poly);

  printf("cModP1 post conversion\n");
  print_gpu<<<1, 1>>>(cModP1.data(), 64, 16);
  cudaDeviceSynchronize();

  cModP1.modp(nBlocks); // cModP1 = rows x nBlocks

  printf("cModP1 post mod\n");
  print_gpu<<<1, 1>>>(cModP1.data(), 64, 16);
  cudaDeviceSynchronize();

  cModP1.bit_transpose(); // cModP1 = mOut x 1

  printf("cModP1 post transpose\n");
  print_gpu<<<1, 1>>>(cModP1.data(), 64, 16);
  cudaDeviceSynchronize();

  xor_gpu<<<16 * mOut / 1024, 1024>>>((uint8_t*) vector.data(), (uint8_t*) cModP1.data(), 16 * mOut);
  cudaDeviceSynchronize();

  Log::end(mRole, CompressTP);
}
