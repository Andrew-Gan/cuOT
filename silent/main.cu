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

void free_multi_gpu(SilentOTSender **sender, SilentOTRecver **recver) {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    delete sender[gpu];
    delete recver[gpu];
  }
}

void base_ot_multi_gpu(SilentOTSender **sender, SilentOTRecver **recver) {
  std::future<void> senderWorker;
  std::future<void> recverWorker;
  for (int gpu = 0; gpu < NGPU; gpu++) {
    senderWorker = std::async([sender, gpu](){
      if (gpu == 0) Log::start(Sender, BaseOT);
      sender[gpu]->base_ot();
      if (gpu == 0) Log::end(Sender, BaseOT);
    });
    recverWorker = std::async([recver, gpu](){
      if (gpu == 0) Log::start(Recver, BaseOT);
      recver[gpu]->base_ot();
      if (gpu == 0) Log::end(Recver, BaseOT);
    });
    senderWorker.get();
    recverWorker.get();
  }
}

void seed_exp_multi_gpu(SilentOT **rcot) {
  Log::start(rcot[0]->mRole, SeedExp);
  std::future<void> worker[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    worker[gpu] = std::async([rcot, gpu](){
      rcot[gpu]->seed_expand();
    });
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    worker[gpu].get();
  }
  Log::end(rcot[0]->mRole, SeedExp);
}

void dual_lpn_multi_gpu(SilentOT **rcot) {
  Log::start(rcot[0]->mRole, LPN);
  std::future<void> worker[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    worker[gpu] = std::async([rcot, gpu](){
      rcot[gpu]->dual_lpn();
    });
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    worker[gpu].get();
  }
  Log::end(rcot[0]->mRole, LPN);
}

int main(int argc, char** argv) {
  int devCount = check_cuda();
  assert(devCount >= NGPU);
  if (argc < 4) {
    fprintf(stderr, "Usage: ./ot protocol logOT numTrees\n");
    return EXIT_FAILURE;
  }
  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);

  std::cout << "logOT: " << logOT << ", trees: " << numTrees << std::endl;
  uint64_t depth = logOT - log2((float) numTrees);
  SilentConfig config = {
    .logOT = logOT,
    .nTree = (uint64_t)numTrees,
    .baseOT = SimplestOT_t,
    .pprf = AesExpand_t,
    .leftKey = {3242342},
    .rightKey = {8993849},
    .dualLPN = QuasiCyclic_t,
  };

  std::stringstream senderFile, recverFile;
  senderFile << "../results/gpu-silent-send-" << logOT << ".txt";
  recverFile << "../results/gpu-silent-recv-" << logOT << ".txt";

  for (int i = 0; i < 2; i++) {
    if(i == 0) std::cout << "initialisation..." << std::endl;
    if(i == 1) std::cout << "benchmarking..." << std::endl;

    SilentOTSender *sender[NGPU];
    SilentOTRecver *recver[NGPU];
    std::future<void> senderWorker[NGPU];
    std::future<void> recverWorker[NGPU];
    config.choices = gen_choices(depth);

    if (i == 1) {
      Log::open(Sender, senderFile.str(), true);
      Log::open(Recver, recverFile.str(), true);
    }

    for (int gpu = 0; gpu < NGPU; gpu++) {
      config.id = gpu;
      sender[gpu] = new SilentOTSender(config);
      recver[gpu] = new SilentOTRecver(config);
    }

    std::cout << "pair init" << std::endl;
    base_ot_multi_gpu(sender, recver);
    std::cout << "pair baseOT" << std::endl;
    seed_exp_multi_gpu((SilentOT**)sender);
    std::cout << "sender exp" << std::endl;
    dual_lpn_multi_gpu((SilentOT**)sender);
    std::cout << "sender lpn" << std::endl;

    for (int gpu = 0; gpu < NGPU; gpu++) {
      recver[gpu]->get_punctured_key();
    }
    seed_exp_multi_gpu((SilentOT**)recver);
    std::cout << "recver exp" << std::endl;
    dual_lpn_multi_gpu((SilentOT**)recver);
    std::cout << "recver lpn" << std::endl;

    free_multi_gpu(sender, recver);
    std::cout << "pair free" << std::endl;

    if (i == 1) {
      Log::close(Sender);
      Log::close(Recver);
    }

    delete[] config.choices;
    // cudaSetDevice(0);
    // Mat recv(recver[gpu]->puncVector[0]);
    // Mat choice(recver[gpu]->choiceVector);
    // assert(check_cot(sender[gpu]->fullVector[0], recv, choice, sender[gpu]->delta[0]));
  }

  std::cout << "all done" << std::endl;

  return EXIT_SUCCESS;
}
