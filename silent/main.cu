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

  for (uint64_t i = 0; i < 2; i++) {
    config.id = i;
    if(i == 0) std::cout << "initialisation..." << std::endl;
    if(i == 1) std::cout << "benchmarking..." << std::endl;
    step = 0;

    std::future<void> senderWorker = std::async(
      [&sender, &step, &config, &bandwidth, &senderFile, i]() {
        if(i == 1) Log::open(Sender, senderFile.str().c_str(), bandwidth, true);
        if(i == 1) Log::start(Sender, CudaInit);
        sender = new SilentOTSender(config);
        if(i == 1) std::cout << "sender init" << std::endl;
        if(i == 1) Log::mem(Sender, CudaInit);
        if(i == 1) Log::end(Sender, CudaInit);
        if(i == 1) Log::start(Sender, BaseOT);
        sender->base_ot();
        if(i == 1) Log::end(Sender, BaseOT);
        if(i == 1) std::cout << "sender baseot" << std::endl;
        if(i == 1) Log::start(Sender, SeedExp);
        sender->pprf_expand();
        if(i == 1) Log::end(Sender, SeedExp);
        if(i == 1) std::cout << "sender seedexp" << std::endl;
        if(i == 1) Log::start(Sender, LPN);
        sender->lpn_compress();
        if(i == 1) Log::end(Sender, LPN);
        if(i == 1) std::cout << "sender lpn" << std::endl;
        if(i == 1) Log::close(Sender);
        step = 1;
      }
    );

    config.choices = gen_choices(depth);

    std::future<void> recverWorker = std::async(
      [&recver, &step, &config, &bandwidth, &recverFile, i]() {
        if(i == 1) Log::open(Recver, recverFile.str().c_str(), bandwidth, true);
        if(i == 1) Log::start(Recver, CudaInit);
        recver = new SilentOTRecver(config);
        if(i == 1) std::cout << "recver init" << std::endl;
        if(i == 1) Log::mem(Recver, CudaInit);
        if(i == 1) Log::end(Recver, CudaInit);
        if(i == 1) Log::start(Recver, BaseOT);
        recver->base_ot();
        if(i == 1) Log::end(Recver, BaseOT);
        if(i == 1) std::cout << "recver baseot" << std::endl;
        while(step < 1);
        while(!recver->expandReady);
        recver->get_punctured_key();
        if(i == 1) Log::start(Recver, SeedExp);
        recver->pprf_expand();
        if(i == 1) Log::end(Recver, SeedExp);
        if(i == 1) std::cout << "recver seedexp" << std::endl;
        if(i == 1) Log::start(Recver, LPN);
        recver->lpn_compress();
        if(i == 1) Log::end(Recver, LPN);
        if(i == 1) std::cout << "recver lpn" << std::endl;
        if(i == 1) Log::close(Recver);
      }
    );

    senderWorker.get();
    recverWorker.get();
    if(i == 1) std::cout << "both done" << std::endl;

    // cudaSetDevice(0);
    // Mat recv(recver->puncVector[0]);
    // Mat choice(recver->choiceVector);
    // assert(check_cot(sender->fullVector[0], recv, choice, sender->delta[0]));

    delete[] config.choices;
    delete sender;
    delete recver;
  }

  cudaDeviceReset();

  return EXIT_SUCCESS;
}
