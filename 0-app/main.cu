#include <stdio.h>
#include <random>
#include <future>
#include <thread>

#include "unit_test.h"
#include "silent_ot.h"

uint64_t* gen_choices(int depth) {
  uint64_t *choices = new uint64_t[depth+1];
  for (int d = 0; d < depth; d++) {
    choices[d] = 0xffffffff;
    // choices[d] = ((uint64_t) rand() << 32) | rand();
  }
  // choice bit for y ^ delta must be invest of final layer
  choices[depth] = ~choices[depth-1];
  return choices;
}

static std::pair<GPUvector<OTblock>, OTblock*> sender_worker(SilentOTConfig config) {
  SilentOTSender ot(config);
  ot.run();
  return ot.get();
}

static std::array<GPUvector<OTblock>, 2> recver_worker(SilentOTConfig config) {
  uint64_t depth = config.logOT - log2((float) config.nTree) + 1;
  uint64_t *choices = gen_choices(depth);
  config.choices = choices;
  SilentOTRecver ot(config);
  ot.run();
  delete[] choices;
  return ot.get();
}

void cuda_init() {
  test_cuda();
  cudaFree(0);
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandDestroyGenerator(prng);
  cufftHandle initPlan;
  cufftCreate(&initPlan);
  cufftDestroy(initPlan);
}

int main(int argc, char** argv) {
  if (argc == 1) {
    test_base_ot();
    test_reduce();
    return 0;
  }
  if (argc < 4) {
    fprintf(stderr, "Usage: ./ot protocol logOT numTrees\n");
    return EXIT_FAILURE;
  }

  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);
  printf("log OTs: %lu, Trees: %d\n", logOT, numTrees);

  // temporary measure while RDMA being set up to run two processes
  char filename[32];
  char filename2[32];
  sprintf(filename, "output/gpu-log-%03d-%03d-send.txt", logOT, numTrees);
  sprintf(filename2, "output/gpu-log-%03d-%03d-recv.txt", logOT, numTrees);
  Log::open(filename, filename2);

  // initialise cuda, curand and cufft
  Log::start(Sender, CudaInit);
  Log::start(Recver, CudaInit);
  cuda_init();
  Log::end(Sender, CudaInit);
  Log::end(Recver, CudaInit);

  SilentOTConfig config = {
    .id = 0, .logOT = logOT, .nTree = numTrees,
    .baseOT = SimplestOT_t,
    .expander = AesHash_t,
    .compressor = QuasiCyclic_t,
  };
  std::future<std::pair<GPUvector<OTblock>, OTblock*>> sender = std::async(sender_worker, config);
  std::future<std::array<GPUvector<OTblock>, 2>> recver = std::async(recver_worker, config);
  auto [fullVector, delta] = sender.get();
  auto [puncVector, choiceVector] = recver.get();

  Log::close();
  
  test_cot(fullVector, delta, puncVector, choiceVector);
  return EXIT_SUCCESS;
}
