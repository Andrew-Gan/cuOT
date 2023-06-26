#include <stdio.h>
#include <random>
#include <future>
#include <thread>

#include "unit_test.h"
#include "silentOT.h"

uint64_t* gen_choices(int numTrees) {
  uint64_t *choices = new uint64_t[numTrees];
  for (int t = 0; t < numTrees; t++) {
    choices[t] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

static std::pair<GPUBlock, GPUBlock> sender_worker(int protocol, int logOT, int numTrees) {
  SilentOTSender ot(0, logOT, numTrees);
  std::pair<GPUBlock, GPUBlock> pair = ot.run();
  return pair;
}

static std::pair<GPUBlock, GPUBlock> recver_worker(int protocol, int logOT, int numTrees) {
  uint64_t *choices = gen_choices(numTrees);
  SilentOTRecver ot(0, logOT, numTrees, choices);
  std::pair<GPUBlock, GPUBlock> pair = ot.run();
  delete[] choices;
  return pair;
}

int main(int argc, char** argv) {
  if (argc == 1) {
    test_aes();
    test_base_ot();
    return 0;
  }
  if (argc < 4) {
    fprintf(stderr, "Usage: ./ot protocol logOT numTrees\n");
    return EXIT_FAILURE;
  }

  test_cuda();
  cudaFree(0);

  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);
  printf("log OTs: %lu, Trees: %d\n", logOT, numTrees);

  // temporary measure while RDMA being set up to run two processes
  char filename[32];
  char filename2[32];
  sprintf(filename, "output/log-%02d-%02d-send.txt", logOT, numTrees);
  sprintf(filename2, "output/log-%02d-%02d-recv.txt", logOT, numTrees);
  EventLog::open(filename, filename2);
  std::future<std::pair<GPUBlock, GPUBlock>> sender = std::async(sender_worker, protocol, logOT, numTrees);
  std::future<std::pair<GPUBlock, GPUBlock>> recver = std::async(recver_worker, protocol, logOT, numTrees);
  auto [fullVector, delta] = sender.get();
  auto [puncVector, choiceVector] = recver.get();
  // test_cot(fullVector, puncVector, choiceVector, delta);
  EventLog::close();
  return EXIT_SUCCESS;
}
