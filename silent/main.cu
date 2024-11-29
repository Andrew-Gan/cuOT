#include <iostream>
#include <sstream>
#include <random>
#include <future>

#include "logger.h"
#include "silent_ot.h"
#include "gpu_tests.h"

#define SAMPLE_SIZE 16

using namespace std;

uint64_t* gen_choices(int depth) {
  uint64_t *choices = new uint64_t[depth];
  for (int d = 0; d < depth; d++) {
    choices[d] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

void silent_ot(vector<SOTSender*>& senders, vector<SOTRecver*>& recvers) {
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

  Log::start(Sender, SeedExp);
  Log::start(Recver, SeedExp);
  vector<future<void>> sT, rT;
  for (SOTSender *s : senders) sT.push_back(async([&s](){s->seed_exp();}));
  for_each(recvers.begin(), recvers.end(), [](auto &r) { r->get_punc_key(); });
  for (SOTRecver *r : recvers) rT.push_back(async([&r](){r->seed_exp();}));
  for_each(sT.begin(), sT.end(), [](auto &w) {w.get();});
  for_each(rT.begin(), rT.end(), [](auto &w) {w.get();});
  Log::end(Sender, SeedExp);
  Log::end(Recver, SeedExp);

  sT.clear();
  rT.clear();

  Log::start(Sender, LPN);
  Log::start(Recver, LPN);
  for (SOTSender *s : senders) sT.push_back(async([&s](){s->dual_lpn();}));
  for (SOTRecver *r : recvers) rT.push_back(async([&r](){r->dual_lpn();}));
  for_each(sT.begin(), sT.end(), [](auto &w) {w.get();});
  for_each(rT.begin(), rT.end(), [](auto &w) {w.get();});
  Log::end(Sender, LPN);
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
  int devCount = check_cuda(2*gpuPerParty);
  assert(devCount >= gpuPerParty);

  printf("logOT: %d, trees: %d, gpus: %d\n", logOT, numTrees, gpuPerParty);
  uint64_t depth = logOT - log2((float) numTrees);
  SilentConfig config = {
    .logOT = logOT, .nTree = (uint64_t)numTrees, .baseOT = SimplestOT_t,
    .pprf = Aes_t, .leftKey = {3242342}, .rightKey = {8993849},
    .dualLPN = QuasiCyclic_t, .gpuPerParty = gpuPerParty,
  };
  config.choices = gen_choices(depth);
  stringstream senderFile, recverFile;
  senderFile << "../results/gpu-sot-send-" << logOT << "-" << gpuPerParty;
  recverFile << "../results/gpu-sot-recv-" << logOT << "-" << gpuPerParty;

  struct timespec start;
  vector<SOTSender*> senders;
  vector<SOTRecver*> recvers;
  for (int gpu = 0; gpu < gpuPerParty; gpu++) {
    config.id = gpu;
    senders.push_back(new SOTSender(config));
    recvers.push_back(new SOTRecver(config));
  }

  for (int i = 0; i < SAMPLE_SIZE+1; i++) {
    // dont benchmark first iteration
    if (i == 1) {
      clock_gettime(CLOCK_MONOTONIC, &start);
      Log::open(Sender, senderFile.str(), SAMPLE_SIZE);
      Log::open(Recver, recverFile.str(), SAMPLE_SIZE);
    }

    silent_ot(senders, recvers);
  }
  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC, &end);
  float elapsed = (end.tv_sec - start.tv_sec) * 1000;
  elapsed += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Average runtime: %.2f\n", elapsed / SAMPLE_SIZE);

  for_each(senders.begin(), senders.end(), [](auto &s) { delete s; });
  for_each(recvers.begin(), recvers.end(), [](auto &r) { delete r; });
  delete[] config.choices;

  Log::close(Sender);
  Log::close(Recver);

  return EXIT_SUCCESS;
}
