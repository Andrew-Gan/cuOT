#include "lpn.h"
#include <cmath>
#include "gpu_tests.h"
#include "gpu_vector.h"
#include "gpu_ops.h"
#include "logger.h"

#define FFT_BATCHSIZE 128

__global__
void bit_to_float(uint64_t *bitPoly, cufftReal *fftReal, uint64_t inBitWidth, uint64_t outRealWidth) {
  uint64_t row = blockIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t tmp = bitPoly[row * (inBitWidth / 64) + col];
  uint64_t offset = row * outRealWidth + 64 * col;
  for (int j = 0; j < 64; j++) {
    fftReal[offset++] = (cufftReal)(tmp & 1);
    tmp >>= 1;
  }
}

__global__
void complex_dot_product(cufftComplex *a_in, cufftComplex *b_io, uint64_t len) {
  uint64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t x = gridDim.x * blockDim.x;
  uint64_t i = blockIdx.y;

  for (int col = j; col < len; col += x) {
    cufftComplex a = a_in[col], b = b_io[i*len+col];
    b.x = a.x * b.x - a.y * b.y;
    b.y = a.x * b.y + a.y * b.x;
    b_io[i*len+col] = b;
  }
}

__global__
void float_to_bit(cufftReal *fftReal, uint64_t *bitPoly, uint64_t mIn, uint64_t scaleLog) {
  uint64_t row = blockIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t setter = 1;
  uint64_t tmp = 0;
  uint64_t offset = row * mIn + 64 * col;
  for (int j = 0; j < 64; j++) {
    if (((int) fftReal[offset++] >> scaleLog) & 1) {
      tmp |= setter;
    }
    setter <<= 1;
  }
  bitPoly[row * (mIn / 64) + col] = tmp;
}

QuasiCyclic::QuasiCyclic(Role role, uint64_t in, uint64_t out) :
  mRole(role), mIn(in), mOut(out) {

  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 1234);
  cufftCreate(&aPlan);
  cufftCreate(&bPlan);
  cufftCreate(&cPlan);
  
  size_t aSize, bSize, cSize;
  cufftMakePlan1d(aPlan, mIn, CUFFT_R2C, 1, &aSize);
  cufftSetAutoAllocation(bPlan, 0);
  cufftSetAutoAllocation(cPlan, 0);
  cufftMakePlan1d(bPlan, mIn, CUFFT_R2C, FFT_BATCHSIZE, &bSize);
  cufftMakePlan1d(cPlan, mIn, CUFFT_C2R, FFT_BATCHSIZE, &cSize);

  cudaMalloc(&workArea, std::max(bSize, cSize));
  cufftSetWorkArea(bPlan, workArea);
  cufftSetWorkArea(cPlan, workArea);
  
  Vec a64(mIn / (8*sizeof(OTblock)));
  curandGenerate(prng, (uint32_t*)a64.data(), 4 * a64.size());
  cudaMalloc(&a64_poly, mIn * sizeof(cufftReal));
  cudaMalloc(&a64_fft, (mIn / 2 + 1) * sizeof(cufftComplex));
  cudaMalloc(&b64_poly, FFT_BATCHSIZE * mIn * sizeof(cufftReal));
  cudaMalloc(&b64_fft, FFT_BATCHSIZE * (mIn / 2 + 1) * sizeof(cufftComplex));
  
  uint64_t thread = mIn / 64;
  uint64_t block = std::min(thread, 1024UL);
  uint64_t grid = (thread + block - 1) / block;
  bit_to_float<<<grid, block>>>((uint64_t*)a64.data(), a64_poly, mIn, mIn);
  cufftExecR2C(aPlan, a64_poly, a64_fft);
  cudaFree(a64_poly);
  cufftDestroy(aPlan);

  uint64_t tmp = mIn;
  while(tmp != 0) {
    tmp >>= 1;
    fftsizeLog++;
  }

  check_call("QuasiCyclic::QuasiCyclic\n");
}

QuasiCyclic::~QuasiCyclic() {
  curandDestroyGenerator(prng);
  cufftDestroy(bPlan);
  cufftDestroy(cPlan);
  cudaFree(a64_fft);
  cudaFree(b64_poly);
  cudaFree(b64_fft);
}

void QuasiCyclic::encode(Vec &vector) {
  Log::mem(mRole, LPN);
  
  Mat b64({mIn, 1});
  b64.clear();
  b64.load((uint8_t*) vector.data(), mOut * sizeof(OTblock));
  b64.bit_transpose();

  // bitpoly to fft
  uint64_t thread1 = mIn / 64;
  uint64_t block1(std::min(thread1, 1024UL));
  dim3 grid1((thread1 + block1 - 1) / block1, FFT_BATCHSIZE);
  // complex dot product and divider
  uint64_t thread2 = (mIn / 2 + 1) / 64;
  uint64_t block2 = std::min(thread2, 1024UL);
  dim3 grid2((thread2 + block2 - 1) / block2, FFT_BATCHSIZE);
  // fft to bitpoly
  uint64_t thread3 = mIn / 64;
  uint64_t block3(std::min(thread3, 1024UL));
  dim3 grid3((thread3 + block3 - 1) / block3, FFT_BATCHSIZE);

  Log::mem(mRole, LPN);

  Mat cModP1({rows, mIn / (8 * sizeof(OTblock))});
  bit_to_float<<<grid1, block1>>>((uint64_t*) b64.data(), b64_poly, mIn, mIn);
  cufftExecR2C(bPlan, b64_poly, b64_fft);
  complex_dot_product<<<grid2, block2>>>(a64_fft, b64_fft, mIn / 2 + 1);
  cufftExecC2R(cPlan, b64_fft, b64_poly);
  float_to_bit<<<grid3, block3>>>(b64_poly, (uint64_t*) cModP1.data(), mIn, fftsizeLog);

  Log::mem(mRole, LPN);

  check_call("QuasiCyclic::fft\n");
  cModP1.modp(mOut / (8 * sizeof(OTblock)));
  cModP1.bit_transpose();
  vector.resize(mOut);
  vector.load((uint8_t*)cModP1.data());

  Log::mem(mRole, LPN);
}
