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

void base_ot_multi_gpu(std::vector<SilentOT*>& senders, std::vector<SilentOT*>& recvers) {
  std::vector<std::future<void>> senderWorkers;
  std::vector<std::future<void>> recverWorkers;
  for (int gpu = 0; gpu < senders.size(); gpu++) {
    senderWorkers.push_back(std::async([senders, gpu](){
      if (gpu == 0) Log::start(Sender, BaseOT);
      senders.at(gpu)->base_ot();
      if (gpu == 0) Log::end(Sender, BaseOT);
    }));
    recverWorkers.push_back(std::async([recvers, gpu](){
      if (gpu == 0) Log::start(Recver, BaseOT);
      recvers.at(gpu)->base_ot();
      if (gpu == 0) Log::end(Recver, BaseOT);
    }));
  }
  for (int gpu = 0; gpu < senders.size(); gpu++) {
    senderWorkers.at(gpu).get();
    recverWorkers.at(gpu).get();
  }
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
      Log::open(Sender, senderFile.str(), true);
      Log::open(Recver, recverFile.str(), true);
    }
    std::vector<SilentOT*> senders;
    std::vector<SilentOT*> recvers;
    config.choices = gen_choices(depth);

    for (int gpu = 0; gpu < gpuPerParty; gpu++) {
      config.id = gpu;
      senders.push_back(new SilentOTSender(config));
      recvers.push_back(new SilentOTRecver(config));
    }

    std::cout << "pair init done" << std::endl;
    base_ot_multi_gpu(senders, recvers);
    std::cout << "pair baseOT done" << std::endl;
    seed_exp_multi_gpu(senders);
    std::cout << "sender exp done" << std::endl;
    dual_lpn_multi_gpu(senders);
    std::cout << "sender lpn done" << std::endl;
    for (int gpu = 0; gpu < gpuPerParty; gpu++) {
      static_cast<SilentOTRecver*>(recvers.at(gpu))->get_punctured_key();
    }
    seed_exp_multi_gpu(recvers);
    std::cout << "recver exp done" << std::endl;
    dual_lpn_multi_gpu(recvers);
    std::cout << "recver lpn done" << std::endl;

    for (int gpu = 0; gpu < gpuPerParty; gpu++) {
      delete senders.at(gpu);
      delete recvers.at(gpu);
    }
    std::cout << "pair free done" << std::endl;

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
