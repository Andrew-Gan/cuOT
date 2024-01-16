#include <cstdio>
#include <random>
#include <future>
#include <thread>

#include "logger.h"
#include "roles.h"
#include "gpu_tests.h"

uint64_t* gen_choices(int depth) {
  uint64_t *choices = new uint64_t[depth+1];
  for (int d = 0; d < depth; d++) {
    choices[d] = ((uint64_t) rand() << 32) | rand();
  }
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
  if (argc < 5) {
    fprintf(stderr, "Usage: ./ot protocol logOT numTrees bandwidth(mbps)\n");
    return EXIT_FAILURE;
  }
  check_cuda();
  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);
  int bandwidth = atoi(argv[4]);
  printf("logOT: %d, numTrees: %d, bandwidth: %d mbps\n", logOT, numTrees, bandwidth);
  uint64_t depth = logOT - log2((float) numTrees) + 1;
  SilentOTConfig config = {
    .id = 0, .logOT = logOT, .nTree = numTrees, .baseOT = SimplestOT_t,
    .expander = AesExpand_t, .leftKey = {3242342}, .rightKey = {8993849},
    .compressor = QuasiCyclic_t,
  };

  SilentOTSender *sender;
  SilentOTRecver *recver;

  char senderFile[60], recverFile[60];
  sprintf(senderFile, "../results/gpu-silent-send-%d-%d-%d.txt", logOT, numTrees, bandwidth);
  sprintf(recverFile, "../results/gpu-silent-recv-%d-%d-%d.txt", logOT, numTrees, bandwidth);
  
  // do not allow simultaneous operation to prevent 2 GPUs on same card from congested PCIe
  std::atomic<int> step = 0;

  std::future<void> senderWorker = std::async([&sender, &step, &config, &bandwidth, &senderFile]() {
    cudaSetDevice(0);
    Log::open(Sender, senderFile, bandwidth, true);
    Log::start(Sender, CudaInit);
    sender = new SilentOTSender(config);
    Log::mem(Sender, CudaInit);
    Log::end(Sender, CudaInit);
    Log::start(Sender, BaseOT);
    sender->base_ot();
    Log::end(Sender, BaseOT);
    Log::start(Sender, SeedExp);
    sender->pprf_expand();
    Log::end(Sender, SeedExp);
    Log::start(Sender, LPN);
    sender->lpn_compress();
    Log::end(Sender, LPN);
    Log::close(Sender);
    step = 1;
  });

  config.choices = gen_choices(depth);

  std::future<void> recverWorker = std::async([&recver, &step, &config, &bandwidth, &recverFile]() {
    cudaSetDevice(1);
    Log::open(Recver, recverFile, bandwidth, true);
    Log::start(Recver, CudaInit);
    recver = new SilentOTRecver(config);
    Log::mem(Recver, CudaInit);
    Log::end(Recver, CudaInit);
    Log::start(Recver, BaseOT);
    recver->base_ot();
    Log::end(Recver, BaseOT);
    while(step < 1);
    while(!recver->expandReady);
    Log::start(Recver, SeedExp);
    recver->pprf_expand();
    Log::end(Recver, SeedExp);
    Log::start(Recver, LPN);
    recver->lpn_compress();
    Log::end(Recver, LPN);
    Log::close(Recver);
  });

  senderWorker.get();
  recverWorker.get();

  delete[] config.choices;
  delete sender;
  delete recver;

  return EXIT_SUCCESS;
}
