#ifndef EMP_LPN_F2K_H__
#define EMP_LPN_F2K_H__

#include "emp-tool/emp-tool.h"

using namespace emp;

//Implementation of local linear code on F_2^k
//Performance highly dependent on the CPU cache size
template<typename IO, int d = 10>
class LpnF2 { public:
	int party;
	int64_t n;
	IO *io;
	int k, mask;
	block seed;
	Mat pubMat;
	int iter = 0;
	bool isInit = false;

	LpnF2 (int party, int64_t n, int k, IO *io) {
		this->party = party;
		this->k = k;
		this->n = n;
		this->io = io;
		mask = 1;
		while(mask < k) {
			mask <<=1;
			mask = mask | 0x1;
		}
	}

	void init(int num_iter) {
		pubMat.resize({(uint64_t)num_iter, (uint64_t)n / 4, (uint64_t)d});

		uint8_t* key_d;
		uint64_t keySize = 11 * AES_KEYLEN;
		PRP prp;
		cuda_malloc((void**)&key_d, keySize * num_iter);
		for (int i = 0; i < num_iter; i++) {
			prp.aes_set_key(seed_gen());
			cuda_memcpy(key_d + i * keySize, prp.aes.rd_key, keySize, H2D);
		}

		cuda_gen_matrices(pubMat, (uint32_t*) key_d);
		cuda_free(key_d);
		isInit = true;
	}

	void compute(Span &nn, Span &kk) {
		if (!isInit) throw std::invalid_argument("Initialise lpn before calling compute()\n");
		int n_0 = nn.size();
		int k_0 = kk.size();

		// bench(nn, kk);
		cuda_lpn_f2_compute(pubMat.data({(uint64_t)iter, 0, 0}), d, n_0, k_0, nn, kk);
	}

	void compute(Vec &nn, Vec &kk, uint64_t consist_check_cot_num) {
		Span nnSpan = nn.span();
		compute(nnSpan, kk, consist_check_cot_num);
	}

	block seed_gen() {
		block seed;
		if(party == ALICE) {
			PRG prg;
			prg.random_block(&seed, 1);
			io->send_data(&seed, sizeof(block));
		} else {
			io->recv_data(&seed, sizeof(block));
		} io->flush();
		return seed;
	}

	void bench(block * nn, const block * kk) {
		// vector<std::future<void>> fut;
		// int64_t width = n/threads;
		// for(int i = 0; i < threads - 1; ++i) {
		// 	int64_t start = i * width;
		// 	int64_t end = min((i+1)* width, n);
		// 	fut.push_back(pool->enqueue([this, nn, kk, start, end]() {
		// 		task(nn, kk, start, end);
		// 	}));
		// }
		// int64_t start = (threads - 1) * width;
        // 	int64_t end = n;
		// task(nn, kk, start, end);

		// for (auto &f: fut) f.get();
	}

};
#endif

// void __compute4(block * nn, const block * kk, int64_t i, PRP * prp) {
// 	block tmp[d];
// 	for(int m = 0; m < d; ++m)
// 		tmp[m] = makeBlock(i, m);
// 	AES_ecb_encrypt_blks(tmp, d, &prp->aes);
// 	// above identical to below:
// 	// prp->permute_block(tmp, d);
// 	uint32_t* r = (uint32_t*)(tmp);
// 	for(int m = 0; m < 4; ++m)
// 		for (int j = 0; j < d; ++j) {
// 			int index = (*r) & mask;
// 			++r;
// 			index = index >= k? index-k:index;
// 			nn[i+m] = nn[i+m] ^ kk[index];
// 		}
// }

// void __compute1(block * nn, const block * kk, int64_t i, PRP*prp) {
// 	const auto nr_blocks = d/4 + (d % 4 != 0);
// 	block tmp[nr_blocks];
// 	for(int m = 0; m < nr_blocks; ++m)
// 		tmp[m] = makeBlock(i, m);
// 	prp->permute_block(tmp, nr_blocks);
// 	uint32_t* r = (uint32_t*)(tmp);
// 	for (int j = 0; j < d; ++j)
// 		nn[i] = nn[i] ^ kk[r[j]%k];
// }

// void task(block * nn, const block * kk, int64_t start, int64_t end) {
// 	PRP prp(seed);
// 	int64_t j = start;
// 	for(; j < end-4; j+=4)
// 		__compute4(nn, kk, j, &prp);
// 	for(; j < end; ++j)
// 		__compute1(nn, kk, j, &prp);
// }
