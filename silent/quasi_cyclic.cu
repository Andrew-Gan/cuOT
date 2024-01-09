#include "compress.h"
#include <cmath>
#include "gpu_tests.h"
#include "gpu_vector.h"
#include "gpu_ops.h"

#include <cstdio>

__global__
void bitpoly_to_cufft(uint64_t *bitPoly, cufftReal *arr) {
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t bitWidth = gridDim.x * blockDim.x;
  uint64_t arrWidth = 2 * 64 * gridDim.x * blockDim.x;
  uint64_t tmp, row = blockIdx.y;
  uint64_t offset = row * arrWidth + 64 * col;

  tmp = bitPoly[row * bitWidth + col];
  for (int j = 0; j < 64; j++) {
    arr[offset++] = tmp & 1;
    tmp >>= 1;
  }
}

__global__
void cufft_to_bitpoly(cufftReal *arr, uint64_t *bitPoly) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t bitWidth = 2 * gridDim.x * blockDim.x;
  uint64_t arrWidth = 2 * 64 * gridDim.x * blockDim.x;
  uint64_t tmp = 0, row = blockIdx.y, col = 64 * i;
  uint64_t offset = row * arrWidth + col;

  uint64_t setter = 1;
  for (int j = 0; j < 64; j++) {
    if ((int) arr[offset++] & 1) {
      tmp |= setter;
      setter <<= 1;
    }
  }
  bitPoly[row * bitWidth + i] = tmp;
}

__global__
void complex_dot_product(cufftComplex *c_out, cufftComplex *a_in, cufftComplex *b_in) {
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t width = gridDim.x * blockDim.x;
  cufftComplex a = a_in[col];
  cufftComplex b, c;

  for (int row = 0; row < 8 * sizeof(OTblock); row++) {
    b = b_in[row * width + col];
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    c_out[row * width + col] = c;
  }
}

QuasiCyclic::QuasiCyclic(uint64_t in, uint64_t out) : mIn(in), mOut(out) {
  if (mIn == 0 || mOut == 0) return;

  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 1234);
  cufftCreate(&aPlan);
  cufftCreate(&bPlan);
  cufftCreate(&cPlan);
  cufftPlan1d(&aPlan, 2 * mIn, CUFFT_R2C, 1);
  cufftPlan1d(&bPlan, 2 * mIn, CUFFT_R2C, rows);
  cufftPlan1d(&cPlan, 2 * mIn, CUFFT_C2R, 1);

  Vec a64(mIn / sizeof(OTblock));
  cufftReal *a64_poly;
  curandGenerate(prng, (uint32_t*)a64.data(), 4 * a64.size());
  cudaMalloc(&a64_poly, 2 * mIn * sizeof(cufftReal));
  cudaMalloc(&a64_fft, 2 * mIn * sizeof(cufftComplex));

  uint64_t thread = 2 * mIn / 64;
  uint64_t block = std::min(thread, 1024lu);
  uint64_t grid = (thread + block - 1) / block;
  bitpoly_to_cufft<<<grid, block>>>((uint64_t*)a64.data(), a64_poly);
  check_call("QuasiCyclic::QuasiCyclic\n");
  cufftExecR2C(aPlan, a64_poly, a64_fft);
  cudaFree(a64_poly);
}

QuasiCyclic::~QuasiCyclic() {
  if (mIn == 0 || mOut == 0) return;
  curandDestroyGenerator(prng);
  cufftDestroy(aPlan);
  cufftDestroy(bPlan);
  cufftDestroy(cPlan);
  cudaFree(a64_fft);
}

void QuasiCyclic::encode(Vec &vector) {
  Mat b64({mIn, 1});
  b64.load((uint8_t*) vector.data());
  b64.bit_transpose();

  cufftReal *b64_poly, *c64_poly;
  cufftComplex *b64_fft, *c64_fft;
  cudaMalloc(&b64_poly, rows * 2 * mIn * sizeof(cufftReal));
  cudaMalloc(&b64_fft, rows * 2 * mIn * sizeof(cufftComplex));
  cudaMalloc(&c64_poly, rows * 2 * mIn * sizeof(cufftReal));
  cudaMalloc(&c64_fft, rows * 2 * mIn * sizeof(cufftComplex));
  check_call("QuasiCyclic::start\n");

  std::cout << "b64: " << std::endl;
  std::cout << b64 << std::endl;

  uint64_t thread = rows * 2 * mIn / 64;
  uint64_t block = std::min(thread, 1024UL);
  uint64_t grid = (thread + block - 1) / block;
  bitpoly_to_cufft<<<grid, block>>>((uint64_t*)b64.data(), b64_poly);
  cufftExecR2C(bPlan, b64_poly, b64_fft);

  uint64_t threadPerRow = 2 * mIn;
  uint64_t block2 = std::min(threadPerRow, 1024UL);
  uint64_t grid2 = (threadPerRow + block - 1) / block;
  complex_dot_product<<<grid2, block2>>>(c64_fft, a64_fft, b64_fft);

  cufftExecC2R(cPlan, c64_fft, c64_poly);
  Mat cModP1({rows, 2 * mIn / sizeof(OTblock)});
  cufft_to_bitpoly<<<grid, block>>>(c64_poly, (uint64_t*) cModP1.data());

  check_call("QuasiCyclic::encode mid\n");

  cudaFree(b64_poly);
  cudaFree(b64_fft);
  cudaFree(c64_poly);
  cudaFree(c64_fft);

  cModP1.modp(mOut / sizeof(OTblock));
  cModP1.bit_transpose();

  vector.resize(mOut);
  vector.load((uint8_t*)cModP1.data());
  check_call("QuasiCyclic::encode end\n");
}
