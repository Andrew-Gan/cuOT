#include "emp-ot/emp-ot.h"
#include "test/test.h"
#include <iostream>
#include <sstream>
#include <future>
#include "../emp-ot/ferret/dev_layer.h"
#include "../emp-ot/ferret/sender.h"
#include "../emp-ot/ferret/recver.h"

enum Phase { NotLastRound, MemcpyRound, LastRound };

void init_multi_gpu(Role role, FerretOT<NetIO> **rcot, FerretConfig config, Mat *data) {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    config.id = gpu;
    switch(role) {
      case Sender:
        rcot[gpu] = new FerretOTSender<NetIO>(config);
        break;
      case Recver:
        rcot[gpu] = new FerretOTRecver<NetIO>(config);
        break;
    }
    rcot[gpu]->rcot_init(data[gpu]);
  }
}

void free_multi_gpu(FerretOT<NetIO> **rcot, FerretConfig config) {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    delete rcot[gpu];
  }
}

void seed_exp_multi_gpu(FerretOT<NetIO> **rcot, Mat *data, Phase phase) {
  Log::start(rcot[0]->mRole, SeedExp);
  std::future<void> worker[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    FerretOT<NetIO> *ot = rcot[gpu];
    Mat *dat = &data[gpu];
    worker[gpu] = std::async([ot, dat, phase]() {
      switch (phase) {
        case NotLastRound: {
          Span pt(*dat, {ot->pt});
          ot->seed_expand(pt);
          break;
        }
        case MemcpyRound:
        case LastRound: {
          Span dataSpan(ot->ot_data);
          ot->seed_expand(dataSpan);
          break;
        }
      }
      sync_dev();
    });
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    worker[gpu].get();
  }
  Log::end(rcot[0]->mRole, SeedExp);
}

void primal_lpn_multi_gpu(FerretOT<NetIO> **rcot, Mat *data, Phase phase) {
  Log::start(rcot[0]->mRole, LPN);
  std::future<void> worker[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    FerretOT<NetIO> *ot = rcot[gpu];
    Mat *dat = &data[gpu];
    worker[gpu] = std::async([ot, dat, phase]() {
      Span pt(*dat, {ot->pt});
      Span dataSpan(ot->ot_data);
      switch (phase) {
        case NotLastRound:
          ot->primal_lpn(pt);
          ot->ot_used = ot->ot_limit;
		      ot->pt += ot->ot_limit;
          break;
        case MemcpyRound:
          ot->primal_lpn(dataSpan);
          memcpy_D2D_dev(pt.data(), ot->ot_data.data(), ot->ot_limit*sizeof(blk));
          ot->pt += ot->ot_limit;
          break;
        case LastRound:
          ot->primal_lpn(dataSpan);
          memcpy_D2D_dev(pt.data(), ot->ot_data.data(), ot->last_round_ot*sizeof(blk));
          break;
      }
      sync_dev();
    });
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    worker[gpu].get();
  }
  Log::end(rcot[0]->mRole, LPN);
}

int main(int argc, char** argv) {
  int port, party;
  Role role = Sender;
	parse_party_and_port(argv, &party, &port);
  switch (party) {
    case 1: role = Sender;
      break;
    case 2: role = Recver;
      break;
  }
	int logOT = argc > 3 ? atoi(argv[3]) : 24;

	std::ostringstream filename;
	filename << "../results/gpu-ferret-";
  if (role == Sender) filename << "send-";
  else filename << "recv-";
  filename << logOT << ".txt";

	NetIO *ios[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    ios[gpu] = new NetIO(role == Sender ? nullptr : "127.0.0.1", port+gpu);
  }

  std::future<void> worker[NGPU];
  FerretConfig config = {
    .logOT = logOT,
    .lpnParam = &ferret_b13,
    .pprf = AesExpand_t,
    .leftKey = {3242342},
    .rightKey = {8993849},
    .primalLPN = PrimalLpn_t,
    .preFile = nullptr,
    .runSetup = true,
    .malicious = false,
  };

  Mat data[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    set_dev(gpu);
    data[gpu].resize({(uint64_t)logOT});
  }

  FerretOT<NetIO> *rcot[NGPU];
	for (int i = 0; i < 2; i++) {
		if (i == 0) std::cout << "initialisation..." << std::endl;
    if (i == 1) std::cout << "benchmarking..." << std::endl;
    if (i == 1) Log::open(role, filename.str(), 1000, true);

    init_multi_gpu(role, rcot, config, data);

    for(int64_t r = 0; r < rcot[0]->round_inplace; r++) {
      seed_exp_multi_gpu(rcot, data, NotLastRound);
      primal_lpn_multi_gpu(rcot, data, NotLastRound);
      for (int gpu = 0; gpu < NGPU; gpu++) {
        rcot[gpu]->ot_used = rcot[gpu]->ot_limit;
        rcot[gpu]->pt += rcot[gpu]->ot_limit;
      }
    }
    if(rcot[0]->round_memcpy) {
      seed_exp_multi_gpu(rcot, data, MemcpyRound);
      primal_lpn_multi_gpu(rcot, data, MemcpyRound);
      for (int gpu = 0; gpu < NGPU; gpu++) {
        rcot[gpu]->pt += rcot[gpu]->ot_limit;
      }
    }
    if(rcot[0]->last_round_ot > 0) {
      seed_exp_multi_gpu(rcot, data, LastRound);
      primal_lpn_multi_gpu(rcot, data, LastRound);
      for (int gpu = 0; gpu < NGPU; gpu++) {
        rcot[gpu]->ot_used = rcot[gpu]->last_round_ot;
      }
    }

    if (i == 1) Log::close(role);
	}
  for (int gpu = 0; gpu < NGPU; gpu++) {
    delete ios[gpu];
  }
}
