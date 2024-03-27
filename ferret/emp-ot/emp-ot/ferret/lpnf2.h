#ifndef __PRIMAL_LPN_H__
#define __PRIMAL_LPN_H__

#include "lpn.h"
#include "gpu_span.h"
#include "dev_layer.h"

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
    PRP prp;
    prp.aes_set_key(seed_gen());
    LpnF2_LpnF2_dev((uint32_t*)prp.aes.rd_key, pubMat);
  }

  void encode(Span &nn, Span &kk) {
    LpnF2_encode_dev(pubMat, n, k, d, nn, kk);
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
