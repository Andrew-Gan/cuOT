#include <iostream>
#include <sstream>
#include <random>
#include <future>

#include "logger.h"
#include "roles.h"
#include "gpu_tests.h"

uint64_t* gen_choices(int depth) {
  uint64_t *choices = new uint64_t[depth];
  for (int d = 0; d < depth; d++) {
    choices[d] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

int main(int argc, char** argv) {
  int devCount = check_cuda();
  assert(devCount >= NGPU);
  if (argc < 5) {
    fprintf(stderr, "Usage: ./ot protocol logOT numTrees bandwidth(mbps)\n");
    return EXIT_FAILURE;
  }
  
  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);
  int bandwidth = atoi(argv[4]);

  std::cout << "logOT: " << logOT << ", trees: " << numTrees << std::endl;
  uint64_t depth = logOT - log2((float) numTrees);
  SilentOTConfig config = {
    .id = 0,
    .logOT = logOT,
    .nTree = (uint64_t)numTrees,
    .baseOT = SimplestOT_t,
    .expander = AesExpand_t,
    .leftKey = {3242342},
    .rightKey = {8993849},
    .compressor = QuasiCyclic_t,
    .ngpuAvail = devCount,
  };

  SilentOTSender *sender;
  SilentOTRecver *recver;

  std::ostringstream senderFile, recverFile;
  senderFile << "../results/gpu-silent-send-" << logOT << "-" << numTrees << ".txt";
  recverFile << "../results/gpu-silent-recv-" << logOT << "-" << numTrees << ".txt";
  
  // prevent simultaneous operation from congesting PCIe
  std::atomic<int> step = 0;

  std::future<void> senderWorker = std::async(
    [&sender, &step, &config, &bandwidth, &senderFile]() {
      Log::open(Sender, senderFile.str().c_str(), bandwidth, true);
      Log::start(Sender, CudaInit);
      sender = new SilentOTSender(config);
      std::cout << "sender init" << std::endl;
      Log::mem(Sender, CudaInit);
      Log::end(Sender, CudaInit);
      Log::start(Sender, BaseOT);
      sender->base_ot();
      Log::end(Sender, BaseOT);
      std::cout << "sender baseot" << std::endl;
      Log::start(Sender, SeedExp);
      sender->pprf_expand();
      Log::end(Sender, SeedExp);
      std::cout << "sender seedexp" << std::endl;
      Log::start(Sender, LPN);
      sender->lpn_compress();
      Log::end(Sender, LPN);
      std::cout << "sender lpn" << std::endl;
      Log::close(Sender);
      step = 1;
    }
  );

  config.choices = gen_choices(depth);

  std::future<void> recverWorker = std::async(
    [&recver, &step, &config, &bandwidth, &recverFile]() {
      Log::open(Recver, recverFile.str().c_str(), bandwidth, true);
      Log::start(Recver, CudaInit);
      recver = new SilentOTRecver(config);
      std::cout << "recver init" << std::endl;
      Log::mem(Recver, CudaInit);
      Log::end(Recver, CudaInit);
      Log::start(Recver, BaseOT);
      recver->base_ot();
      Log::end(Recver, BaseOT);
      std::cout << "recver baseot" << std::endl;
      while(step < 1);
      while(!recver->expandReady);
      Log::start(Recver, SeedExp);
      recver->pprf_expand();
      Log::end(Recver, SeedExp);
       std::cout << "recver seedexp" << std::endl;
      Log::start(Recver, LPN);
      recver->lpn_compress();
      Log::end(Recver, LPN);
      std::cout << "recver lpn" << std::endl;
      Log::close(Recver);
    }
  );

  senderWorker.get();
  recverWorker.get();

  std::cout << "both done" << std::endl;

  return 0;

  // cudaSetDevice(0);
  // Mat recv(recver->puncVector[0]);
  // Mat choice(recver->choiceVector);
  // assert(check_cot(sender->fullVector[0], recv, choice, sender->delta[0]));

  delete[] config.choices;
  delete sender;
  delete recver;

  return EXIT_SUCCESS;
}
