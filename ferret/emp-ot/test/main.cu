#include "emp-ot/emp-ot.h"
#include "test/test.h"
#include "../emp-ot/ferret/cuda_layer.h"
#include <iostream>
#include <sstream>
#include <future>
#include "roles.h"

using namespace std;

int main(int argc, char** argv) {
  int port, party;
	parse_party_and_port(argv, &party, &port);
	int logOT = argc > 3 ? atoi(argv[3]) : 24;

	std::ostringstream filename;
	filename << "../results/gpu-ferret-";
  if (party == Sender) filename << "send-";
  else filename << "recv-";
  filename << logOT << ".txt";

	NetIO ios[NGPU];
  for (int gpu = 0; gpu < NGPU) {
    ios[gpu] = new NetIO(party == Sender ? nullptr : "127.0.0.1", port+gpu);
  }

  FerretOT *ot[NGPU];
  std::future<void> worker[NGPU];
  FerretConfig config = {
    .logOT = logOT,
    .lpnParam = ferret_b13,
    .pprf = AesExpand_t,
    .leftKey = {3242342},
    .rightKey = {8993849},
    .primalLPN = ferretLPN,
    .runSetup = true,
  };

	for (int i = 0; i < 2; i++) {
		if(i == 0) std::cout << "initialisation..." << std::endl;
    if(i == 1) std::cout << "benchmarking..." << std::endl;
    for (int gpu = 0; gpu < NGPU; gpu++) {
      worker[gpu] = std::async([&]() {
        cudaSetDevice(gpu);
        Log::open(party, filename, 1000, true);
        ot[gpu] = new FerretOTSender(config);
        Mat b({length});
        io->sync();
        ot[gpu]->rcot(b);
        Log::close(party);
      });
    }
    for (int gpu = 0; gpu < NGPU; gpu++) {
      worker[gpu].get();
    }
	}
}
