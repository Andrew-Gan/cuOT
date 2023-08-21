#include <stdio.h>
#include <random>
#include <future>
#include <thread>

#include "unit_test.h"
#include "silent_ot.h"

uint64_t* gen_choices(int depth) {
  uint64_t *choices = new uint64_t[depth+1];
  for (int d = 0; d < depth; d++) {
    choices[d] = ((uint64_t) rand() << 32) | rand();
  }
  // choice bit for y ^ delta must be invest of final layer
  choices[depth] = ~choices[depth-1];
  return choices;
}

void cuda_init() {
  cudaFree(0);
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandDestroyGenerator(prng);
  cufftHandle initPlan;
  cufftCreate(&initPlan);
  cufftDestroy(initPlan);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: ./ot protocol logOT numTrees\n");
    return EXIT_FAILURE;
  }
  test_cuda();

  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);
  printf("log OTs: %lu, Trees: %d\n", logOT, numTrees);

  // temporary measure while RDMA being set up to run two processes
  char filenameS[32], filenameR[32];
  sprintf(filenameS, "output/gpu-log-%03d-%03d-send.txt", logOT, numTrees);
  sprintf(filenameR, "output/gpu-log-%03d-%03d-recv.txt", logOT, numTrees);

  uint64_t depth = logOT - log2((float) numTrees) + 1;

  SilentOTConfig config = {
    .id = 0, .logOT = logOT, .nTree = numTrees,
    .baseOT = SimplestOT_t,
    .expander = AesHash_t,
    .compressor = QuasiCyclic_t,
    .choices = gen_choices(depth),
  };

  SilentOTSender *sender;
  SilentOTRecver *recver;

  std::future<void> senderWorker = std::async([&sender, &config, filenameS]() {
    // Log::start(Sender, CudaInit);
    cudaSetDevice(0);
    cuda_init();
    // Log::end(Sender, CudaInit);
    Log::open(Sender, filenameS);
    sender = new SilentOTSender(config);
    sender->run();
    Log::close(Sender);
  });

  std::future<void> recverWorker = std::async([&recver, &config, filenameR]() {
    // Log::start(Recver, CudaInit);
    cudaSetDevice(1);
    cuda_init();
    // Log::end(Recver, CudaInit);
    Log::open(Recver, filenameR);
    recver = new SilentOTRecver(config);
    recver->run();
    Log::close(Recver);
  });

  senderWorker.get();
  recverWorker.get();

  // comment out when profiling
  // test_cot(*sender, *recver);

  delete[] config.choices;
  delete sender;
  delete recver;

  return EXIT_SUCCESS;
}
