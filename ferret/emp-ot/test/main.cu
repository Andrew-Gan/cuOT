#include "emp-ot/emp-ot.h"
#include "test/test.h"
#include "../emp-ot/ferret/cuda_layer.h"
#include <iostream>
#include <sstream>
#include <future>
#include "../emp-ot/ferret/roles.h"

enum Phase { NotLastRound, Memcpy, LastRound };

void init_multi_gpu(Role role, FerretOT **rcot, FerretConfig config, Mat *data) {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    config.id = gpu;
    switch(role) {
      case Sender:
        rcot[gpu] = new FerretOTSender(config);
        break;
      case Recver:
        rcot[gpu] = new FerretOTRecver(config);
        break;
    }

    io->sync();
    rcot[gpu]->rcot_init(data[gpu]);
  }
}

void free_multi_gpu(FerretOT **rcot, FerretConfig config) {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    delete rcot[gpu];
  }
}

void seed_exp_multi_gpu(FerretOT **rcot, Phase phase) {
  Log::start(rcot[0]->mRole, SeedExp);
  Span *output[NGPU];
  std::future<void> worker[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    Mat *ot = rcot[gpu];
    worker[gpu] = std::async([ot, gpu, output]() {
      switch (phase) {
        case Iterations: output = new Span(ot->data, pt);
          break;
        case Memcpy:
        case LastRound: output = new Span(ot->ot_data);
          break;
      }
      ot->send_preot();
      ot->seed_expand(*output);
    });
    cudaDeviceSynchronize();
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    worker[gpu].get();
    delete output[gpu];
  }
  Log::end(rcot[0]->mRole, SeedExp);
}

void primal_lpn_multi_gpu(FerretOT **rcot, Mat *data, Phase phase) {
  Log::start(rcot[0]->mRole, LPN);
  Span *output[NGPU];
  std::future<void> worker[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    Mat *ot = rcot[gpu];
    worker[gpu] = std::async([ot, gpu, output, phase, data]() {
      switch (phase) {
        case Iterations: output = new Span(ot->data, ot->pt);
          break;
        case Memcpy:
        case LastRound: output = new Span(ot->ot_data);
          break;
      }
      ot->primal_lpn(*output);
      ot->ot_used = 0;
      size_t cpySize = 0;
      switch (phase) {
        case Memcpy: cpySize = ot->ot_limit;
          break;
        case LastRound: cpySize = ot->last_round_ot;
          break;
      }
      if (phase == Memcpy || phase == LastRound) {
        cudaMemcpy(data.data(ot->pt), ot->ot_data.data(), cpySize*sizeof(blk), cudaMemcpyDeviceToDevice);
      }
      cudaDeviceSynchronize();
    });
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    worker[gpu].get();
    delete output[gpu];
  }
  Log::end(rcot[0]->mRole, LPN);
}

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

  Mat *data[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    data[gpu].resize({logOT});
  }

  FerretOT *ot[NGPU];
	for (int i = 0; i < 2; i++) {
		if(i == 0) std::cout << "initialisation..." << std::endl;
    if(i == 1) std::cout << "benchmarking..." << std::endl;
    if (i == 1) Log::open(party, filename, 1000, true);

    init_multi_gpu(party, ot, config, data);

    for(uint64_t i = 0; i < round_inplace; ++i) {
      seed_exp_multi_gpu(rcot, NotLastRound);
      primal_lpn_multi_gpu(rcot, data, NotLastRound);
      for (int gpu = 0; gpu < NGPU; gpu++) {
        rcot[gpu]->ot_used = rcot[gpu]->ot_limit;
        rcot[gpu]->pt += rcot[gpu]->ot_limit;
      }
    }
    if(rcot[gpu]->round_memcpy) {
      seed_exp_multi_gpu(rcot, Memcpy);
      primal_lpn_multi_gpu(rcot, data, Memcpy);
      for (int gpu = 0; gpu < NGPU; gpu++) {
        rcot[gpu]->pt += rcot[gpu]->ot_limit;
      }
    }
    if(last_round_ot > 0) {
      seed_exp_multi_gpu(rcot, LastRound);
      primal_lpn_multi_gpu(rcot, data, LastRound);
      for (int gpu = 0; gpu < NGPU; gpu++) {
        rcot[gpu]->ot_used = rcot[gpu]->last_round_ot;
      }
    }

    if (i == 1) Log::close(party);
	}
}
