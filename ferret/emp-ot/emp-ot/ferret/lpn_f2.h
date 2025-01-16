#ifndef EMP_LPN_F2K_H__
#define EMP_LPN_F2K_H__

#include "dev_layer.h"
#include "emp-tool/emp-tool.h"
using namespace emp;

//Implementation of local linear code on F_2^k
//Performance highly dependent on the CPU cache size
template<typename IO, int d = 10>
class LpnF2 { public:
	int party;
	int64_t n;
	IO *io;
	int ngpu;
	int nPerGPU;
	int k, mask;
	block seed;
	ThreadPool *pool;
	Mat *pubMats;

	LpnF2 (int party, int64_t n, int k, IO *io, ThreadPool *pool, int ngpu) {
		this->party = party;
		this->k = k;
		this->n = n;
		this->io = io;
		this->ngpu = ngpu;
		this->pool = pool;
		this->nPerGPU = n / ngpu;
		mask = 1;
		while(mask < k) {
			mask <<=1;
			mask = mask | 0x1;
		}
		pubMats = new Mat[ngpu];
		GPU_PARALLEL_FOR(
			pubMats[i].resize({((uint64_t)nPerGPU * d + 3) / 4});
		)
	}

	virtual ~LpnF2() {
		delete[] pubMats;
	}

	void compute(Mat *nn, blk **kk) {
		std::cout << std::endl;
		vector<std::future<void>> fut;
		seed = seed_gen();
		PRP prp(seed);
		uint32_t *key = (uint32_t*)prp.aes.rd_key;
		GPU_PARALLEL_FOR(
			cuda_primal_lpn(pubMats[i], d, nPerGPU, k, key, nn[i], kk[i]);
		)
	}

	block seed_gen() {
		block seed;
		if(party == ALICE) {
			PRG prg;
			prg.random_block(&seed, 1);
			io->send_data(&seed, sizeof(block));
		} else {
			io->recv_data(&seed, sizeof(block));
		}
		io->flush();
		return seed;
	}
};
#endif
