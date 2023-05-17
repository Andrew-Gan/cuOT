#include <stdio.h>
#include <random>
#include <future>

#include "unit_test.h"
#include "silent_ot.h"

uint64_t* gen_choices(int numTrees) {
  uint64_t *choices = new uint64_t[numTrees];
  for (int t = 0; t < numTrees; t++) {
    choices[t] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

static void sender_worker(OT *ot) {
  GPUBlock m0, m1;
  ot->send(m0, m1);
}

static void recver_worker(OT *ot, int numTrees) {
  uint64_t *choices = gen_choices(numTrees);
  GPUBlock mb = dynamic_cast<SilentOT*>(ot)->recv(choices);
  delete[] choices;
}

int main(int argc, char** argv) {
  if (argc == 1) {
    test_cuda();
    test_aes();
    test_base_ot();
    return 0;
  }
  if (argc < 5) {
    fprintf(stderr, "Usage: ./ot protocol depth trees logfile\n");
    return EXIT_FAILURE;
  }

  int protocol = atoi(argv[1]);
  int logOT = atoi(argv[2]);
  int numTrees = atoi(argv[3]);
  printf("log OTs: %lu, Trees: %d\n", logOT, numTrees);

  OT *ot_send, *ot_recv;

  EventLog::open(argv[4]);
  switch (protocol) {
    case 1:
      ot_send = new SilentOT(Sender, 0, logOT, numTrees);
      ot_recv = new SilentOT(Recver, 0, logOT, numTrees);
      break;
  }

  std::future sender = std::async(sender_worker, ot_send);
  std::future recver = std::async(recver_worker, ot_recv, numTrees);
  sender.get();
  recver.get();

  EventLog::close();
  delete ot_send;
  delete ot_recv;
  return EXIT_SUCCESS;
}
