#include <stdio.h>
#include <random>
#include <future>
#include <thread>

#include "unit_test.h"
#include "silent_ot.h"

uint64_t* gen_choices(int numTrees) {
  uint64_t *choices = new uint64_t[numTrees];
  for (int t = 0; t < numTrees; t++) {
    choices[t] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

static void sender_worker(int protocol, int logOT, int numTrees) {
  SilentOT *ot;
  switch(protocol) {
    case 1: ot = new SilentOT(Sender, 0, logOT, numTrees);
      break;
  }
  GPUBlock m0, m1;
  ot->send(m0, m1);
  delete ot;
}

static void recver_worker(int protocol, int logOT, int numTrees) {
  SilentOT *ot;
  switch(protocol) {
    case 1: ot = new SilentOT(Recver, 0, logOT, numTrees);
      break;
  }
  uint64_t *choices = gen_choices(numTrees);
  GPUBlock mb = ot->recv(choices);
  delete[] choices;
  delete ot;
}

int main(int argc, char** argv) {
  if (argc == 1) {
    test_aes();
    test_base_ot();
    return 0;
  }
  if (argc < 4) {
    fprintf(stderr, "Usage: ./ot protocol depth trees\n");
    return EXIT_FAILURE;
  }

  test_cuda();

  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);
  printf("log OTs: %lu, Trees: %d\n", logOT, numTrees);

  EventLog::open("log.txt");
  std::future sender = std::async(sender_worker, protocol, logOT, numTrees);
  std::future recver = std::async(recver_worker, protocol, logOT, numTrees);
  sender.get();
  recver.get();
  EventLog::close();
  return EXIT_SUCCESS;
}
