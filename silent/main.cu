#include <iostream>
#include <sstream>
#include <random>
#include <future>

#include "logger.h"
#include "roles.h"
#include "gpu_tests.h"

#define SAMPLE_SIZE 8

uint64_t* gen_choices(int depth) {
  uint64_t *choices = new uint64_t[depth];
  for (int d = 0; d < depth; d++) {
    choices[d] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

void base_ot_multi_gpu(std::vector<SilentOT*>& senders, std::vector<SilentOT*>& recvers) {
  std::vector<std::future<void>> workers;
  workers.push_back(std::async([senders](){
    Log::start(Sender, BaseOT);
    senders.at(0)->base_ot();
    Log::end(Sender, BaseOT);
  }));
  workers.push_back(std::async([recvers](){
    Log::start(Recver, BaseOT);
    recvers.at(0)->base_ot();
    Log::end(Recver, BaseOT);
  }));
  workers.at(0).get();
  workers.at(1).get();
}

void seed_exp_multi_gpu(std::vector<SilentOT*>& rcots) {
  Log::start(rcots.at(0)->mRole, SeedExp);
  std::vector<std::future<void>> workers;
  for (SilentOT *rcot : rcots) {
    workers.push_back(std::async([rcot](){
      rcot->seed_expand();
    }));
  }
  for (auto &t : workers) {
    t.get();
  }
  Log::end(rcots.at(0)->mRole, SeedExp);
}

void dual_lpn_multi_gpu(std::vector<SilentOT*>& rcots) {
  cudaSetDevice(0);
  Log::start(rcots.at(0)->mRole, LPN);
  std::vector<std::future<void>> workers;
  for (SilentOT *rcot : rcots) {
    workers.push_back(std::async([rcot](){
      rcot->dual_lpn();
    }));
  }
  for (auto &t : workers) {
    t.get();
  }
  Log::end(rcots.at(0)->mRole, LPN);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: ./ot logOT numTrees gpuPerParty\n");
    return EXIT_FAILURE;
  }
  int logOT = atoi(argv[1]);
  int numTrees = atoi(argv[2]);
  int gpuPerParty = atoi(argv[3]);

  int devCount = check_cuda();
  assert(devCount >= gpuPerParty);

  std::cout << "logOT: " << logOT << ", trees: " << numTrees << ", gpus: " << gpuPerParty << std::endl;
  uint64_t depth = logOT - log2((float) numTrees);
  SilentConfig config = {
    .logOT = logOT,
    .nTree = (uint64_t)numTrees,
    .baseOT = SimplestOT_t,
    .pprf = Aes_t,
    .leftKey = {3242342},
    .rightKey = {8993849},
    .dualLPN = QuasiCyclic_t,
    .gpuPerParty = gpuPerParty,
  };

  std::stringstream senderFile, recverFile;
  senderFile << "../results/gpu-silent-send-" << logOT << "-" << gpuPerParty << ".txt";
  recverFile << "../results/gpu-silent-recv-" << logOT << "-" << gpuPerParty << ".txt";

  for (int i = 0; i < SAMPLE_SIZE+1; i++) {
    // dont benchmark first iteration
    if (i == 1) {
      Log::open(Sender, senderFile.str(), true, SAMPLE_SIZE);
      Log::open(Recver, recverFile.str(), true, SAMPLE_SIZE);
    }
    std::vector<SilentOT*> senders;
    std::vector<SilentOT*> recvers;
    config.choices = gen_choices(depth);

    for (int gpu = 0; gpu < gpuPerParty; gpu++) {
      config.id = gpu;
      senders.push_back(new SilentOTSender(config));
      recvers.push_back(new SilentOTRecver(config));
    }

    base_ot_multi_gpu(senders, recvers);
    seed_exp_multi_gpu(senders);
    dual_lpn_multi_gpu(senders);
    // for (int gpu = 0; gpu < gpuPerParty; gpu++) {
    //   static_cast<SilentOTRecver*>(recvers.at(gpu))->get_punctured_key();
    // }
    // seed_exp_multi_gpu(recvers);
    // dual_lpn_multi_gpu(recvers);

    for (int gpu = 0; gpu < gpuPerParty; gpu++) {
      delete senders.at(gpu);
      delete recvers.at(gpu);
    }

    delete[] config.choices;
    // cudaSetDevice(0);
    // Mat recv(recver[gpu]->puncVector[0]);
    // Mat choice(recver[gpu]->choiceVector);
    // assert(check_cot(sender[gpu]->fullVector[0], recv, choice, sender[gpu]->delta[0]));
  }
  Log::close(Sender);
  Log::close(Recver);

  std::cout << "all done" << std::endl;

  return EXIT_SUCCESS;
}
