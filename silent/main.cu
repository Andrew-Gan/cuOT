#include <iostream>
#include <sstream>
#include <random>
#include <future>

#include "logger.h"
#include "roles.h"
#include "gpu_tests.h"

#define SAMPLE_SIZE 8

using namespace std;

uint64_t* gen_choices(int depth) {
  uint64_t *choices = new uint64_t[depth];
  for (int d = 0; d < depth; d++) {
    choices[d] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

void base_ot(vector<SOTSender*>& senders, vector<SOTRecver*>& recvers) {
  vector<future<void>> workers;
  workers.push_back(async([&senders](){
    Log::start(Sender, BaseOT);
    senders.at(0)->base_ot();
    Log::end(Sender, BaseOT);
  }));
  workers.push_back(async([&recvers](){
    Log::start(Recver, BaseOT);
    recvers.at(0)->base_ot();
    Log::end(Recver, BaseOT);
  }));
  workers.at(0).get();
  workers.at(1).get();
}

void seed_expansion(vector<SOTSender*>& senders, vector<SOTRecver*>& recvers) {
  Log::start(Sender, SeedExp);
  vector<future<void>> workers;
  for (SOTSender *sender : senders) {
    workers.push_back(async([&sender](){ sender->seed_expand(); }));
  }
  for_each(workers.begin(), workers.end(), [](auto &w) { w.get(); });
  Log::end(Sender, SeedExp);

  Log::start(Recver, SeedExp);
  workers.clear();
  for (SOTRecver *recver : recvers) {
    workers.push_back(async([&recver](){ recver->seed_expand(); }));
  }
  for_each(workers.begin(), workers.end(), [](auto &w) { w.get(); });
  Log::end(Recver, SeedExp);
}

void dual_lpn(vector<SOTSender*>& senders, vector<SOTRecver*>& recvers) {
  Log::start(Sender, LPN);
  vector<future<void>> workers;
  for (SOTSender *sender : senders) {
    workers.push_back(async([&sender](){ sender->dual_lpn(); }));
  }
  for_each(workers.begin(), workers.end(), [](auto &w) { w.get(); });
  Log::end(Sender, LPN);

  Log::start(Recver, LPN);
  workers.clear();
  for (SOTRecver *recver : recvers) {
    workers.push_back(async([&recver](){ recver->dual_lpn(); }));
  }
  for_each(workers.begin(), workers.end(), [](auto &w) { w.get(); });
  Log::end(Recver, LPN);
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

  printf("logOT: %d, trees: %d, gpus: %d\n", logOT, numTrees, gpuPerParty);
  uint64_t depth = logOT - log2((float) numTrees);
  SilentConfig config = {
    .logOT = logOT, .nTree = (uint64_t)numTrees, .baseOT = SimplestOT_t,
    .pprf = Aes_t, .leftKey = {3242342}, .rightKey = {8993849},
    .dualLPN = QuasiCyclic_t, .gpuPerParty = gpuPerParty,
  };
  stringstream senderFile, recverFile;
  senderFile << "../results/gpu-sot-send-" << logOT << "-" << gpuPerParty;
  recverFile << "../results/gpu-sot-recv-" << logOT << "-" << gpuPerParty;

  for (int i = 0; i < SAMPLE_SIZE+1; i++) {
    // dont benchmark first iteration
    if (i == 1) {
      Log::open(Sender, senderFile.str(), true, SAMPLE_SIZE);
      Log::open(Recver, recverFile.str(), true, SAMPLE_SIZE);
    }
    vector<SOTSender*> senders;
    vector<SOTRecver*> recvers;
    config.choices = gen_choices(depth);

    for (int gpu = 0; gpu < gpuPerParty; gpu++) {
      config.id = gpu;
      senders.push_back(new SOTSender(config));
      recvers.push_back(new SOTRecver(config));
    }

    base_ot(senders, recvers);
    seed_expansion(senders, recvers);
    dual_lpn(senders, recvers);
    for_each(recvers.begin(), recvers.end(), [](auto &r) { r->get_punc_key(); });

    delete[] config.choices;
    // cudaSetDevice(0);
    // Mat recv(recver[gpu]->puncVector[0]);
    // Mat choice(recver[gpu]->choiceVector);
    // assert(check_cot(sender[gpu]->fullVector[0], recv, choice, sender[gpu]->delta[0]));

    for_each(senders.begin(), senders.end(), [](auto &s) { delete s; });
    for_each(recvers.begin(), recvers.end(), [](auto &r) { delete r; });
  }
  Log::close(Sender);
  Log::close(Recver);

  return EXIT_SUCCESS;
}
