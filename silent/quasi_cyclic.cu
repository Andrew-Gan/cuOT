#include "compress.h"
#include <cmath>
#include "gpu_tests.h"
#include "gpu_vector.h"
#include "gpu_ops.h"

#define BATCH_SIZE 8

__global__
void bitpoly_to_cufft(uint64_t *bitPoly, cufftReal *fftReal, uint64_t inBitWidth, uint64_t outRealWidth) {
  uint64_t row = blockIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t tmp = bitPoly[row * (inBitWidth / 64) + col];
  uint64_t offset = row * outRealWidth + 64 * col;
  for (int j = 0; j < 64; j++) {
    fftReal[offset++] = tmp & 1;
    tmp >>= 1;
  }
}

__global__
void complex_dot_product(cufftComplex *c_out, cufftComplex *a_in, cufftComplex *b_in) {
  uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t x = gridDim.x * blockDim.x;
  uint64_t i = blockIdx.y;
  cufftComplex a = a_in[j], b = b_in[i*x+j], c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  c_out[i] = c;
}

__global__
void divider(cufftReal *data, int scale, uint64_t realWidth) {
    uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    data[i * realWidth + j] /= scale;
}

__global__
void cufft_to_bitpoly(cufftReal *fftReal, uint64_t *bitPoly, uint64_t inRealWidth, uint64_t outBitWidth) {
  uint64_t row = blockIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t setter = 1;
  uint64_t tmp = 0;
  uint64_t offset = row * inRealWidth + 64 * col;
  for (int j = 0; j < 64; j++) {
    if ((int) fftReal[offset++] & 1) {
      tmp |= setter;
    }
    setter <<= 1;
  }
  bitPoly[row * (outBitWidth / 64) + col] = tmp;
}

QuasiCyclic::QuasiCyclic(uint64_t in, uint64_t out) : mIn(in), mOut(out) {
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 1234);
  cufftCreate(&aPlan);
  cufftCreate(&bPlan);
  cufftCreate(&cPlan);
  int fftsize = 2 * mIn;
  cufftPlan1d(&aPlan, fftsize, CUFFT_R2C, 1);
  cufftPlanMany(&bPlan, 1, &fftsize, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, BATCH_SIZE);
  cufftPlanMany(&cPlan, 1, &fftsize, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, BATCH_SIZE);

  Vec a64(mIn / sizeof(OTblock));
  cufftReal *a64_poly;
  curandGenerate(prng, (uint32_t*)a64.data(), 4 * a64.size());

  cudaMalloc(&a64_poly, 2 * mIn * sizeof(cufftReal));
  cudaMalloc(&a64_fft, 2 * mIn * sizeof(cufftComplex));

  uint64_t thread = mIn / 64;
  uint64_t block = std::min(thread, 1024UL);
  uint64_t grid = (thread + block - 1) / block;
  bitpoly_to_cufft<<<grid, block>>>((uint64_t*)a64.data(), a64_poly, mIn, 2 * mIn);

  check_call("QuasiCyclic::QuasiCyclic\n");
  cufftExecR2C(aPlan, a64_poly, a64_fft);
  cudaFree(a64_poly);
}

QuasiCyclic::~QuasiCyclic() {
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
  cudaMalloc(&b64_poly, BATCH_SIZE * 2 * mIn * sizeof(cufftReal));
  cudaMalloc(&b64_fft, BATCH_SIZE * 2 * mIn * sizeof(cufftComplex));
  cudaMalloc(&c64_poly, BATCH_SIZE * 2 * mIn * sizeof(cufftReal));
  cudaMalloc(&c64_fft, BATCH_SIZE * 2 * mIn * sizeof(cufftComplex));
  check_call("QuasiCyclic::malloc\n");

  // bitpoly to cufftReal
  uint64_t thread1 = mIn / 64;
  uint64_t block1(std::min(thread1, 1024UL));
  dim3 grid1((thread1 + block1 - 1) / block1, BATCH_SIZE);
  // complex dot product and divider
  uint64_t thread2 = 2 * mIn;
  uint64_t block2 = std::min(thread2, 1024UL);
  dim3 grid2((thread2 + block2 - 1) / block2, BATCH_SIZE);
  // cufftReal to bitpoly
  uint64_t thread3 = 2 * mIn / 64;
  uint64_t block3(std::min(thread3, 1024UL));
  dim3 grid3((thread3 + block3 - 1) / block3, BATCH_SIZE);

  Mat cModP1({rows, (2 * mIn) / (8 * sizeof(OTblock))});

  for (uint64_t i = 0; i < rows; i += BATCH_SIZE) {
    bitpoly_to_cufft<<<grid1, block1>>>((uint64_t*) b64.data({i, 0}), b64_poly, mIn, 2 * mIn);
    cufftExecR2C(bPlan, b64_poly, b64_fft);
    complex_dot_product<<<grid2, block2>>>(c64_fft, a64_fft, b64_fft);
    cufftExecC2R(cPlan, c64_fft, c64_poly);
    divider<<<grid2, block2>>>(c64_poly, 2 * mIn, 2 * mIn);
    cufft_to_bitpoly<<<grid3, block3>>>(c64_poly, (uint64_t*) cModP1.data({i, 0}), 2 * mIn, 2 * mIn);
  }
  check_call("QuasiCyclic::fft\n");

  cudaFree(b64_poly);
  cudaFree(b64_fft);
  cudaFree(c64_poly);
  cudaFree(c64_fft);

  cModP1.modp(mOut / (8 * sizeof(OTblock)));
  cModP1.bit_transpose();

  vector.resize(mOut);
  vector.load((uint8_t*)cModP1.data());
  check_call("QuasiCyclic::ifft\n");
}
