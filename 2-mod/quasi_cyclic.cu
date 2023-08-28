#include "compressor.h"
#include <cmath>
#include "gpu_vector.h"
#include "gpu_ops.h"

// rows to run FFT at once: 1-128
#define FFT_BATCHSIZE 8

__global__
void bitPolyToCufftArray(uint64_t *bitPoly, cufftReal *arr, uint64_t rows) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t n64 = gridDim.x * blockDim.x;
  uint64_t arrWidth = 2 * 64 * gridDim.x * blockDim.x;
  uint64_t tmp, col;

  for (uint64_t r = 0; r < rows; r++) {
    tmp = bitPoly[r * n64 + i];
    for (uint64_t j = 0; j < 64; j++) {
      col = 64 * i + j;
      arr[r * arrWidth + col] = tmp & (1 << j) ? 1 : 0;
    }
  }
}

__global__
void cufftArrayToBitPoly(cufftReal *arr, uint64_t *bitPoly, uint64_t rows) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t n64 = gridDim.x * blockDim.x;
  uint64_t fftSize = 2 * 64 * n64;
  uint64_t arrWidth = 2 * 64 * gridDim.x * blockDim.x;
  uint64_t tmp, col;

  for (uint64_t r = 0; r < rows; r++) {
    for (uint64_t j = 0; j < 64; j++) {
      col = 64 * i + j;
      if ((int) arr[r * arrWidth + col] % 2)
        tmp |= 1 << j;
      else
        tmp &= ~(1 << j);
    }
    bitPoly[r * n64 + i] = tmp;
  }
}

__global__
void complex_dot_product(cufftComplex *c_out, cufftComplex *a_in, cufftComplex *b_in) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t nT = gridDim.x * blockDim.x;
  uint64_t width = 2 * nT;
  cufftComplex a[2] = { a_in[tid], a_in[tid + nT] };
  cufftComplex b, c;

  for (uint64_t row = 0; row < FFT_BATCHSIZE; row++) {
    b = b_in[row * width + tid];
    c.x = a[0].x * b.x - a[0].y * b.y;
    c.y = a[0].x * b.y + a[0].y * b.x;
    c_out[row * width + tid] = c;

    b = b_in[row * width + tid + nT];
    c.x = a[1].x * b.x - a[1].y * b.y;
    c.y = a[1].x * b.y + a[1].y * b.x;
    c_out[row * width + tid + nT] = c;
  }
}

QuasiCyclic::QuasiCyclic(Role role, uint64_t in, uint64_t out) : mRole(role), mIn(in), mOut(out) {
  if (mIn == 0 || mOut == 0) return;
  
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 50);
  nBlocks = (mOut + rows - 1) / rows;
  n2Blocks = ((mIn - mOut) + rows - 1) / rows;
  n64 = nBlocks * 2;

  cufftCreate(&aPlan);
  cufftCreate(&bPlan);
  cufftCreate(&cPlan);
  cufftPlan1d(&aPlan, 2 * mOut, CUFFT_R2C, 1);
  cufftPlan1d(&bPlan, 2 * mOut, CUFFT_R2C, FFT_BATCHSIZE);
  cufftPlan1d(&cPlan, 2 * mOut, CUFFT_C2R, FFT_BATCHSIZE);

  GPUvector<uint64_t> a64(n64);
  cufftReal *a64_poly;
  curandGenerate(prng, (uint32_t*) a64.data(), 2 * n64);

  cudaMalloc(&a64_poly, 2 * mOut * sizeof(cufftReal));
  cudaMalloc(&a64_fft, 2 * mOut * sizeof(cufftComplex));

  uint64_t blk = std::min(n64, 1024lu);
  uint64_t grid = n64 < 1024 ? 1 : n64 / 1024;
  bitPolyToCufftArray<<<grid, blk>>>(a64.data(), a64_poly, 1);
  cudaDeviceSynchronize();

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

void QuasiCyclic::encode(GPUvector<OTblock> &vector) {
  // XT = mOut x 1
  GPUmatrix<OTblock> XT(mOut, 1);
  XT.load((uint8_t*) (vector.data() + mOut));
  // XT = rows x n2blocks
  XT.bit_transpose();

  // XT.load("input/XT.bin");

  uint64_t *b64 = (uint64_t*) XT.data();
  cufftReal *b64_poly, *c64_poly;
  cufftComplex *b64_fft, *c64_fft;
  cudaMalloc(&b64_poly, FFT_BATCHSIZE * 2 * mOut * sizeof(cufftReal));
  cudaMalloc(&b64_fft, FFT_BATCHSIZE * 2 * mOut * sizeof(cufftComplex));
  cudaMalloc(&c64_poly, FFT_BATCHSIZE * 2 * mOut * sizeof(cufftReal));
  cudaMalloc(&c64_fft, FFT_BATCHSIZE * 2 * mOut * sizeof(cufftComplex));

  GPUmatrix<OTblock> cModP1(rows, 2 * nBlocks); // hold unmodded coeffs
  uint64_t grid, blk;

  for (uint64_t r = 0; r < rows; r += FFT_BATCHSIZE) {
    blk = std::min(n64, 1024lu);
    grid = n64 < 1024 ? 1 : n64 / 1024;
    bitPolyToCufftArray<<<grid, blk>>>(b64 + r * n64, b64_poly, FFT_BATCHSIZE);
    cufftExecR2C(bPlan, b64_poly, b64_fft);

    blk = std::min(mOut, 1024lu);
    grid = mOut < 1024 ? 1 : mOut / 1024;
    complex_dot_product<<<grid, blk>>>(c64_fft, a64_fft, b64_fft);

    cufftExecC2R(cPlan, c64_fft, c64_poly);
    blk = std::min(n64, 1024lu);
    grid = n64 < 1024 ? 1 : n64 / 1024;
    cufftArrayToBitPoly<<<grid, blk>>>(c64_poly, (uint64_t*) cModP1.data() + r * 2 * n64, FFT_BATCHSIZE);
  }

  cudaFree(b64_poly);
  cudaFree(b64_fft);
  cudaFree(c64_poly);
  cudaFree(c64_fft);

  cModP1.modp(nBlocks); // cModP1 = rows x nBlocks
  cModP1.bit_transpose(); // cModP1 = mOut x 1

  xor_gpu<<<16 * mOut / 1024, 1024>>>((uint8_t*) vector.data(), (uint8_t*) cModP1.data(), 16 * mOut);
  cudaDeviceSynchronize();
}
