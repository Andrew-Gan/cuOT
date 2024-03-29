#include "emp-ot/emp-ot.h"
#include "test/test.h"
#include <iostream>
#include <sstream>
#include <future>
#include "../emp-ot/ferret/dev_layer.h"
#include "../emp-ot/ferret/sender.h"
#include "../emp-ot/ferret/recver.h"

enum Phase { NotLastRound, MemcpyRound, LastRound };

void init_multi_gpu(Role role, FerretOT<NetIO> **rcot, FerretConfig config, Mat *data, NetIO **ios) {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    config.id = gpu;
    switch(role) {
      case Sender:
        rcot[gpu] = new FerretOTSender<NetIO>(config, ios[gpu]);
        break;
      case Recver:
        rcot[gpu] = new FerretOTRecver<NetIO>(config, ios[gpu]);
        break;
    }
    rcot[gpu]->rcot_init(data[gpu]);
  }
}

void free_multi_gpu(FerretOT<NetIO> **rcot) {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    delete rcot[gpu];
  }
}

void seed_exp_multi_gpu(FerretOT<NetIO> **rcot, Mat *data, Phase phase) {
  Log::start(rcot[0]->mRole, SeedExp);
  std::future<void> worker[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    FerretOT<NetIO> *ot = rcot[0];
    Mat *dat = &data[0];
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
          ot->ot_used = ot->last_round_ot;
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
	filename << "../../results/gpu-ferret-";
  if (role == Sender) filename << "send-";
  else filename << "recv-";
  filename << logOT << ".txt";

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
  NetIO *ios[NGPU];

  for (int gpu = 0; gpu < NGPU; gpu++) {
    set_dev(gpu);
    data[gpu].resize({(uint64_t)(1<<logOT)});
    ios[gpu] = new NetIO(role==Sender?nullptr:"127.0.0.1",port+gpu);
  }

  FerretOT<NetIO> *rcot[NGPU];
  std::string partyName(role==Sender?"Sender":"Recver");

  Log::open(role, filename.str(), 1000, true);

  init_multi_gpu(role, rcot, config, data, ios);
  std::cout << "pair init" << std::endl;

  for(int64_t r = 0; r < rcot[0]->round_inplace; r++) {
    seed_exp_multi_gpu(rcot, data, NotLastRound);
    std::cout << partyName << " iter expand" << std::endl;
    primal_lpn_multi_gpu(rcot, data, NotLastRound);
    std::cout << partyName << " iter lpn" << std::endl;
  }
  if(rcot[0]->round_memcpy) {
    seed_exp_multi_gpu(rcot, data, MemcpyRound);
    std::cout << partyName << " memcpy expand" << std::endl;
    primal_lpn_multi_gpu(rcot, data, MemcpyRound);
    std::cout << partyName << " memcpy lpn" << std::endl;
  }
  if(rcot[0]->last_round_ot > 0) {
    seed_exp_multi_gpu(rcot, data, LastRound);
    std::cout << partyName << " last expand" << std::endl;
    primal_lpn_multi_gpu(rcot, data, LastRound);
    std::cout << partyName << " last lpn" << std::endl;
  }

  free_multi_gpu(rcot);

  Log::close(role);
}
