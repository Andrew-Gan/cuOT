#include "compressor.h"
#include <cmath>
#include "gpu_vector.h"
#include "gpu_ops.h"

// rows to run FFT at once: 1-128
#define FFT_BATCHSIZE 8

__global__
void bitPolyToCufftArray(uint64_t *bitPoly, cufftReal *arr) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint64_t j = 0; j < 64; j++) {
    arr[64 * i + j] = bitPoly[i] & (1 << j);
  }
}

__global__
void cufftArrayToBitPoly(cufftReal *arr, uint64_t *bitPoly) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint64_t j = 0; j < 64; j++) {
    if ((int)(arr[64 * i + j]) % 2)
      bitPoly[i] |= 1 << j;
    else
      bitPoly[i] &= ~(1 << j);
  }
}

__global__
void complex_dot_product(cufftComplex *c_out, cufftComplex *a_in, cufftComplex *b_in) {
  uint64_t row = blockIdx.y;
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t width = gridDim.x * blockDim.x;

  c_out[row * width + tid].x = a_in[tid].x * b_in[row * width + tid].x - (a_in[tid].y * b_in[row * width + tid].y);
  c_out[row * width + tid].y = a_in[tid].x * b_in[row * width + tid].y + (a_in[tid].y * b_in[row * width + tid].x);
}

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
  cufftPlan1d(&aPlan, 2 * mOut, CUFFT_R2C, 1);
  cufftPlan1d(&bPlan, 2 * mOut, CUFFT_R2C, FFT_BATCHSIZE);
  cufftPlan1d(&cPlan, 2 * mOut, CUFFT_C2R, FFT_BATCHSIZE);

  GPUvector<uint64_t> a64(n64);
  cufftReal *a64_poly;
  curandGenerate(prng, (uint32_t*) a64.data(), 2 * n64);

  // // take in next prime
  // a64.resize(16386);
  // a64.load("input/a64_poly.bin");
  // a64.save("output/a64_poly.bin");

  // std::ofstream ofs("output/a64_poly.txt");
  // ofs << a64 << std::endl;
  // ofs.close();

  cudaMalloc(&a64_poly, 2 * mOut * sizeof(cufftReal));
  cudaMalloc(&a64_fft, 2 * mOut * sizeof(cufftComplex));
  Log::end(mRole, CompressInit);

  Log::start(mRole, CompressFFT);
  uint64_t blk = std::min(n64, 1024lu);
  uint64_t grid = n64 < 1024 ? 1 : n64 / 1024;
  bitPolyToCufftArray<<<grid, blk>>>((uint64_t*) a64.data(), a64_poly);
  cudaDeviceSynchronize();

  cufftExecR2C(aPlan, a64_poly, a64_fft);
  cudaFree(a64_poly);
  Log::end(mRole, CompressFFT);

  // cufftComplex *buffer = new cufftComplex[n64];
  // cudaMemcpy(buffer, a64_fft, n64 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  // ofs.open("output/a64_fft.txt");
  // for (int i = 0; i < n64; i++) pipeline{
  //   ofs << buffer[i].x << std::endl;
  // }
  // ofs.close();
  // delete[] buffer;
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
  // XT = mOut x 1
  GPUmatrix<OTblock> XT(mOut, 1);
  XT.load((uint8_t*) (vector.data() + mOut));
  // XT = rows x n2blocks
  XT.bit_transpose();
  Log::end(mRole, CompressTP);

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
  dim3 grid2;

  for (uint64_t r = 0; r < XT.rows() / FFT_BATCHSIZE; r++) {
    blk = std::min(FFT_BATCHSIZE * n64, 1024lu);
    grid = n64 < 1024 ? 1 : FFT_BATCHSIZE * n64 / 1024;

    Log::start(mRole, CompressFFT);
    bitPolyToCufftArray<<<grid, blk>>>(b64 + (r * FFT_BATCHSIZE * n64), b64_poly);
    cudaDeviceSynchronize();
    cufftExecR2C(bPlan, b64_poly, b64_fft);
    Log::end(mRole, CompressFFT);

    Log::start(mRole, CompressMult);
    blk = std::min(2 * mOut, 1024lu);
    grid2 = dim3(2 * mOut < 1024 ? 1 : 2 * mOut / 1024, FFT_BATCHSIZE, 1);
    complex_dot_product<<<grid2, blk>>>(c64_fft, a64_fft, b64_fft);
    cudaDeviceSynchronize();
    Log::end(mRole, CompressMult);

    Log::start(mRole, CompressIFFT);
    cufftExecC2R(cPlan, c64_fft, c64_poly);
    blk = std::min(FFT_BATCHSIZE * n64, 1024lu);
    grid = n64 < 1024 ? 1 : FFT_BATCHSIZE * n64 / 1024;
    cufftArrayToBitPoly<<<grid, blk>>>(c64_poly, (uint64_t*) cModP1.data());
    cudaDeviceSynchronize();
    Log::end(mRole, CompressIFFT);
  }

  cudaFree(b64_poly);
  cudaFree(b64_fft);
  cudaFree(c64_poly);
  cudaFree(c64_fft);

  Log::start(mRole, CompressTP);
  cModP1.modp(nBlocks); // cModP1 = rows x nBlocks
  cModP1.bit_transpose(); // cModP1 = mOut x 1

  xor_gpu<<<16 * mOut / 1024, 1024>>>((uint8_t*) vector.data(), (uint8_t*) cModP1.data(), 16 * mOut);
  cudaDeviceSynchronize();

  Log::end(mRole, CompressTP);
}
