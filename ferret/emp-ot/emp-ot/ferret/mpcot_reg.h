#ifndef MPCOT_REG_H__
#define MPCOT_REG_H__

#include <emp-tool/emp-tool.h>
#include "emp-ot/ferret/preot.h"

#include "dev_layer.h"

using namespace emp;
using std::future;

template<typename IO>
class MpcotReg {
public:
	int party;
	int ngpu;
	int item_n, idx_max, m, tPerGPU;
	int tree_height, leave_n;
	int tree_n;
	int consist_check_cot_num;
	bool is_malicious;
	ThreadPool *pool;
	PRG prg;
	IO *netio;
	IO **ios;
	block Delta_f2k;
	block *consist_check_chi_alpha = nullptr, *consist_check_VW = nullptr;

	// prevent runtime malloc
	Mat *buffer;
	Mat *separated;
	
	std::vector<uint32_t> item_pos_recver;
	GaloisFieldPacking pack;

	MpcotReg(int party, int ngpu, int n, int t, int log_bin_sz, ThreadPool *pool, IO **ios) {
		this->party = party;
		this->ngpu = ngpu;
		netio = ios[0];
		this->ios = ios;
		consist_check_cot_num = 128;
		this->pool = pool;
		this->is_malicious = false;

		this->item_n = t;
		this->idx_max = n;
		this->tree_height = log_bin_sz+1;
		this->leave_n = 1<<(this->tree_height-1);
		this->tree_n = this->item_n;
		this->tPerGPU = (t + (ngpu - 1)) / ngpu;

		buffer = new Mat[ngpu];
		separated = new Mat[ngpu];

		GPU_PARALLEL_FOR(
			buffer[i].resize({tPerGPU * (1UL << log_bin_sz)});
			separated[i].resize({tPerGPU * (1UL << log_bin_sz)});
		)
	}

	virtual ~MpcotReg() {
		delete[] buffer;
		delete[] separated;
	}

	void set_malicious() {
		this->is_malicious = true;
	}

	void sender_init(block delta) {
		Delta_f2k = delta;
	}

	void recver_init() {
		item_pos_recver.resize(this->item_n);
	}

	// MPFSS F_2k
	void mpcot(Mat *sparse_vector, OTPre<IO> *ot, Mat *pre_cot_data) {
		// if(party == BOB) consist_check_chi_alpha = new block[item_n];
		// consist_check_VW = new block[item_n];

		if(party == ALICE) {
			mpcot_init_sender(ot);
			exec_parallel_sender(ot, sparse_vector);
		} else {
			bool *choice = new bool[item_n * (tree_height-1)];
			mpcot_init_recver(choice, ot);
			exec_parallel_recver(ot, sparse_vector, choice);
			delete[] choice;
		}

		// if(is_malicious)
		// 	consistency_check_f2k(pre_cot_data, item_n);

		// for (auto p : senders) delete p;
		// for (auto p : recvers) delete p;

		// if(party == BOB) delete[] consist_check_chi_alpha;
		// delete[] consist_check_VW;
	}

	void mpcot_init_sender(OTPre<IO> *ot) {
		for(int i = 0; i < item_n; ++i) {
			ot->choices_sender();
		}
		netio->flush();
		ot->reset();
	}

	void mpcot_init_recver(bool *choice, OTPre<IO> *ot) {
		for(int t = 0; t < item_n; ++t) {
			ot->choices_recver(choice+t*(tree_height-1));
			item_pos_recver[t] = 0;
			for(int i = 0; i < tree_height-1; ++i) {
				item_pos_recver[i] <<= 1;
				if(!choice[t*ot->length+i])
					item_pos_recver[i] += 1;
			}
		}
		netio->flush();
		ot->reset();
	}

	void exec_parallel_sender(OTPre<IO> *ot, Mat *sparse_vector) {
		blk *delta = (blk*)&Delta_f2k;
		vector<future<void>> fut;
		GPU_PARALLEL_FOR(
			blk *m0 = new blk[tPerGPU*(tree_height-1)];
			blk *m1 = new blk[tPerGPU*(tree_height-1)];
			blk *secret = new blk[tPerGPU];
			cuda_mpcot_sender(sparse_vector[i], buffer[i], separated[i],
				m0, m1, secret, tPerGPU, tree_height-1, delta);
			for (int t = 0; t < tPerGPU; t++) {
				block *lSum = (block*)m0 + t * (tree_height-1);
				block *rSum = (block*)m1 + t * (tree_height-1);
				ot->send(lSum, rSum, tree_height-1, ios[i], i * tPerGPU + t);
			}

			ios[i]->send_data(secret, tPerGPU * sizeof(blk));
			ios[i]->flush();
			delete[] m0;
			delete[] m1;
			delete[] secret;
		)
	}

	void exec_parallel_recver(OTPre<IO> *ot, Mat *sparse_vector, bool *choice) {
		vector<future<void>> fut;
		GPU_PARALLEL_FOR(
			blk *mc = new blk[tPerGPU*(tree_height-1)];
			blk *secret = new blk[tPerGPU];
			for (int t = 0; t < tPerGPU; t++) {
				block *cSum = (block*)mc + t * (tree_height-1);
				bool *c = &choice[(i * tPerGPU + t) * (tree_height-1)];
				ot->recv(cSum, c, tree_height-1, ios[i], i * tPerGPU + t);
			}
			ios[i]->recv_data(secret, tPerGPU * sizeof(blk));
			cuda_mpcot_recver(sparse_vector[i], buffer[i], separated[i],
				mc, secret, tPerGPU, tree_height-1, choice);
			delete[] mc;
			delete[] secret;
		)
	}

	// f2k consistency check
	void consistency_check_f2k(block *pre_cot_data, int num) {
		if(this->party == ALICE) {
			block r1, r2;
			vector_self_xor(&r1, this->consist_check_VW, num);
			bool x_prime[128];
			this->netio->recv_data(x_prime, 128*sizeof(bool));
			for(int i = 0; i < 128; ++i) {
				if(x_prime[i])
					pre_cot_data[i] = pre_cot_data[i] ^ this->Delta_f2k;
			}
			pack.packing(&r2, pre_cot_data);
			r1 = r1 ^ r2;
			block dig[2];
			Hash hash;
			hash.hash_once(dig, &r1, sizeof(block));
			this->netio->send_data(dig, 2*sizeof(block));
			this->netio->flush();
		} else {
			block r1, r2, r3;
			vector_self_xor(&r1, this->consist_check_VW, num);
			vector_self_xor(&r2, this->consist_check_chi_alpha, num);
			uint64_t pos[2];
			pos[0] = _mm_extract_epi64(r2, 0);
			pos[1] = _mm_extract_epi64(r2, 1);
			bool pre_cot_bool[128];
			for(int i = 0; i < 2; ++i) {
				for(int j = 0; j < 64; ++j) {
					pre_cot_bool[i*64+j] = ((pos[i] & 1) == 1) ^ getLSB(pre_cot_data[i*64+j]);
					pos[i] >>= 1;
				}
			}
			this->netio->send_data(pre_cot_bool, 128*sizeof(bool));
			this->netio->flush();
			pack.packing(&r3, pre_cot_data);
			r1 = r1 ^ r3;
			block dig[2];
			Hash hash;
			hash.hash_once(dig, &r1, sizeof(block));
			block recv[2];
			this->netio->recv_data(recv, 2*sizeof(block));
			if(!cmpBlock(dig, recv, 2))
				std::cout << "SPCOT consistency check fails" << std::endl;
		}
	}
};
#endif
