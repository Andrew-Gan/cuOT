#include <cstdio>
#include <random>
#include <future>
#include <thread>

#include "event_log.h"
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
  if (argc < 4) {
    fprintf(stderr, "Usage: ./ot protocol logOT numTrees\n");
    return EXIT_FAILURE;
  }
  check_cuda();
  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);
  printf("log OTs: %lu, Trees: %d\n", logOT, numTrees);
  uint64_t depth = logOT - log2((float) numTrees) + 1;
  SilentOTConfig config = {
    .id = 0,
    .logOT = logOT,
    .nTree = numTrees,
    .baseOT = SimplestOT_t,
    .expander = AesExpand_t,
    .compressor = QuasiCyclic_t,
    .choices = gen_choices(depth),
  };

  SilentOTSender *sender;
  SilentOTRecver *recver;
  Log::open(Sender, "../results/gpu-silent-send.txt");
  Log::open(Recver, "../results/gpu-silent-recv.txt");

  std::future<void> senderWorker = std::async([&sender, &config]() {
    cudaSetDevice(0);
    sender = new SilentOTSender(config);
    Log::start(Sender, BaseOT);
    sender->base_ot();
    Log::end(Sender, BaseOT);
  });
  std::future<void> recverWorker = std::async([&recver, &config]() {
    cudaSetDevice(1);
    recver = new SilentOTRecver(config);
    Log::start(Recver, BaseOT);
    recver->base_ot();
    Log::end(Recver, BaseOT);
  });
  senderWorker.get();
  recverWorker.get();

  std::vector<Vec> &l = sender->leftHash;
  std::vector<Vec> &r = sender->rightHash;
  std::vector<Vec> &c = recver->choiceHash;
  for (int i = 0; i < c.size(); i++) {
    assert(check_rot(l.at(i), r.at(i), c.at(i), config.choices[i]));
  }
  std::cout << "ROT test successful" << std::endl;

  senderWorker = std::async([&sender, &config]() {
    cudaSetDevice(0);
    Log::start(Sender, SeedExp);
    sender->pprf_expand();
    Log::end(Sender, SeedExp);
    Log::close(Sender);
  });
  recverWorker = std::async([&recver, &config]() {
    cudaSetDevice(1);
    Log::start(Recver, SeedExp);
    recver->pprf_expand();
    Log::end(Recver, SeedExp);
    Log::close(Recver);
  });
  senderWorker.get();
  recverWorker.get();
  assert(check_cot(sender->fullVector, recver->puncVector, recver->choiceVector, sender->delta));
  std::cout << "pre-LPN COT test successful" << std::endl;

  senderWorker = std::async([&sender, &config]() {
    cudaSetDevice(0);
    Log::start(Sender, LPN);
    sender->lpn_compress();
    Log::end(Sender, LPN);
    Log::close(Sender);
  });
  recverWorker = std::async([&recver, &config]() {
    cudaSetDevice(1);
    Log::start(Recver, LPN);
    recver->lpn_compress();
    Log::end(Recver, LPN);
    Log::close(Recver);
  });
  senderWorker.get();
  recverWorker.get();

  assert(check_cot(sender->fullVector, recver->puncVector, recver->choiceVector, sender->delta));
  std::cout << "post-LPN COT test successful" << std::endl;

  delete[] config.choices;
  delete sender;
  delete recver;

  return EXIT_SUCCESS;
}
