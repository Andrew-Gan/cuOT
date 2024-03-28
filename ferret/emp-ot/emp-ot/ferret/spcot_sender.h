#ifndef SPCOT_SENDER_H__
#define SPCOT_SENDER_H__
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "emp-ot/emp-ot.h"
#include "emp-ot/ferret/twokeyprp.h"
#include "gpu_span.h"
#include "pprf.h"

#include "dev_layer.h"

using namespace emp;

template<typename IO>
class SPCOT_Sender {
public:
  blk seed;
  blk *delta;
  Span *ggm_tree;
  Mat lSum, rSum;
  IO *io;
  uint64_t depth, leave_n;
  PRG prg;
  blk secret_sum_f2;
  uint64_t tree_n;

  SPCOT_Sender(IO *io, uint64_t tree_n, uint64_t depth_in) {
    this->tree_n = tree_n;
    initialization(io, depth_in);
    block seed128;
    prg.random_block(&seed128, 1);
    memcpy(&seed, &seed128, sizeof(block));
  }

  void initialization(IO *io, uint64_t depth_in) {
    this->io = io;
    this->depth = depth_in;
    this->leave_n = 1<<(this->depth);
    lSum.resize({depth-1, tree_n});
    rSum.resize({depth-1, tree_n});
  }
  
  // generate GGM tree, transfer secret, F2^k
  void compute(Span &tree, blk *secret) {
    this->ggm_tree = &tree;
    this->delta = secret;
    
    uint32_t k0_blk[4] = {3242342};
    uint32_t k1_blk[4] = {8993849};
    Mat buffer({tree.size()});
    Span bufferSpan(buffer);
    Span *input = &bufferSpan;
    Span *output = &tree;
    AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
    Mat separated({tree.size()});

    for (uint64_t d = 0, inWidth = 1; d < depth-1; d++, inWidth *= 2) {
      std::swap(input, output);
      aesExpand.expand(*input, *output, separated, tree_n * inWidth);
      separated.sum(2 * tree_n, inWidth);
      memcpy_D2D_dev(lSum.data({d, 0}), separated.data({0}), tree_n * sizeof(blk));
      memcpy_D2D_dev(rSum.data({d, 0}), separated.data({tree_n}), tree_n * sizeof(blk));
    }
    if (output != &tree) {
      memcpy_D2D_dev(tree.data(), output->data(), tree.size_bytes());
    }
  }

  // send the nodes by oblivious transfer, F2^k
  template<typename OT>
  void send_f2k(OT * ot, IO * io2) {
    block *lSum_cpu = new block[tree_n*(depth-1)];
    block *rSum_cpu = new block[tree_n*(depth-1)];

    memcpy_D2H_dev(lSum_cpu, lSum.data(), tree_n*(depth-1)*sizeof(blk));
    memcpy_D2H_dev(rSum_cpu, rSum.data(), tree_n*(depth-1)*sizeof(blk));

    ot->send(lSum_cpu, rSum_cpu, tree_n*(depth-1), io2, 0);
    io2->send_data(&secret_sum_f2, sizeof(blk));

    delete[] lSum_cpu;
    delete[] rSum_cpu;
  }

  void consistency_check_msg_gen(block *V) {
    // X
    // block *chi = new block[leave_n];
    // Hash hash;
    // block digest[2];
    // hash.hash_once(digest, &secret_sum_f2, sizeof(block));
    // uni_hash_coeff_gen(chi, digest[0], leave_n);

    // vector_inn_prdt_sum_red(V, chi, ggm_tree, leave_n);
    // delete[] chi;
  }
};

#endif
