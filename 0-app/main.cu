#include <stdio.h>
#include <random>
#include <future>
#include <thread>

#include "unit_test.h"
#include "silent_ot.h"

uint64_t* gen_choices(int numTrees) {
  uint64_t *choices = new uint64_t[numTrees];
  for (int t = 0; t < numTrees; t++) {
    choices[t] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

static std::pair<GPUBlock, GPUBlock> sender_worker(int protocol, int logOT, int numTrees) {
  SilentOT *ot;
  switch(protocol) {
    case 1: ot = new SilentOT(Sender, 0, logOT, numTrees);
      break;
  }
  std::pair<GPUBlock, GPUBlock> pair = ot->send();
  delete ot;
  return pair;
}

static std::pair<GPUBlock, SparseVector> recver_worker(int protocol, int logOT, int numTrees) {
  SilentOT *ot;
  switch(protocol) {
    case 1: ot = new SilentOT(Recver, 0, logOT, numTrees);
      break;
  }
  uint64_t *choices = gen_choices(numTrees);
  std::pair<GPUBlock, SparseVector> pair = ot->recv(choices);
  delete[] choices;
  delete ot;
  return pair;
}

int main(int argc, char** argv) {
  if (argc == 1) {
    test_aes();
    test_base_ot();
    return 0;
  }
  if (argc < 5) {
    fprintf(stderr, "Usage: ./ot protocol logOT trees logfile\n");
    return EXIT_FAILURE;
  }

  test_cuda();
  cudaFree(0);

  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);
  printf("log OTs: %lu, Trees: %d\n", logOT, numTrees);

  EventLog::open(argv[4]);
  std::future sender = std::async(sender_worker, protocol, logOT, numTrees);
  std::future recver = std::async(recver_worker, protocol, logOT, numTrees);
  auto [fullVector, delta] = sender.get();
  auto [puncVector, choiceVector] = recver.get();
  test_cot(fullVector, puncVector, choiceVector, delta);
  EventLog::close();
  return EXIT_SUCCESS;
}
