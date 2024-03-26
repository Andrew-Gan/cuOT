#ifndef __PRIMAL_LPN_H__
#define __PRIMAL_LPN_H__

#include "gpu_span.h"
#include "lpn.h"
#include "aes_op.h"

__global__
void make_block(blk *blocks) {
	int x = blockDim.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    blocks[i*x+j].data[0] = 4*i;
    blocks[i*x+j].data[2] = j;
}

__global__
void lpn_single_row(uint32_t *r, int d, int k, blk *nn, blk *kk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    blk tmp_nn = nn[i];
	blk tmp_kk;
    for (int j = 0; j < d; j++) {
        tmp_kk = kk[r[i*d+j] % k];
        tmp_nn.data[0] ^= tmp_kk.data[0];
        tmp_nn.data[1] ^= tmp_kk.data[1];
        tmp_nn.data[2] ^= tmp_kk.data[2];
        tmp_nn.data[3] ^= tmp_kk.data[3];
    }
    nn[i] = tmp_nn;
}

//Implementation of local linear code on F_2^k
//Performance highly dependent on the GPU shared memory size
template<typename IO>
class LpnF2 : PrimalLpn { 
public:
  Role party;
  uint64_t n;
  IO *io;
  uint64_t k;
  int mask;
  block seed;
  Mat pubMat;
  const uint64_t d = 10; // random matrix density

  LpnF2(Role party, uint64_t n, uint64_t k, IO *io) {
    this->party = party;
    this->k = k;
    this->n = n;
    this->io = io;
    mask = 1;
    while(mask < k) {
      mask <<= 1;
      mask = mask | 0x1;
    }
    pubMat.resize({n / 4, d});
    uint32_t* key;
    uint64_t keySize = 11 * AES_KEYLEN;
    PRP prp;
    cudaMalloc(&key, keySize);
    prp.aes_set_key(seed_gen());
    cudaMemcpy(key, prp.aes.rd_key, keySize, cudaMemcpyHostToDevice);
    dim3 grid(pubMat.dim(1), pubMat.dim(0) / 1024);
    dim3 block(1, 1024);
    make_block<<<grid, block>>>(pubMat.data());
    uint64_t grid2 = 4*pubMat.dim(0)*pubMat.dim(1)/AES_BSIZE;
    aesEncrypt128<<<grid2, AES_BSIZE>>>(key, (uint32_t*)pubMat.data());
    cudaDeviceSynchronize();
    cudaFree(key);
  }

  void encode(Span &nn, Span &kk) {
    int n_0 = nn.size();
    int k_0 = kk.size();
    lpn_single_row<<<n/1024, 1024>>>((uint32_t*)pubMat.data(), d, k, nn.data(), kk.data());
    cudaDeviceSynchronize();
  }

  void encode(Mat &nn, Mat &kk) {
    Span nnSpan(nn);
    encode(nnSpan, kk);
  }

  block seed_gen() {
    block seed;
    if(party == ALICE) {
      PRG prg;
      prg.random_block(&seed, 1);
      io->send_data(&seed, sizeof(block));
    } else {
      io->recv_data(&seed, sizeof(block));
    } io->flush();
    return seed;
  }
};

#endif
