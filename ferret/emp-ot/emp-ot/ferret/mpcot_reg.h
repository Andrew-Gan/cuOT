#ifndef MPCOT_REG_H__
#define MPCOT_REG_H__

#include <emp-tool/emp-tool.h>
#include <set>
#include "emp-ot/ferret/spcot_sender.h"
#include "emp-ot/ferret/spcot_recver.h"
#include "emp-ot/ferret/preot.h"

#include "dev_layer.h"
#include "logger.h"

using namespace emp;
using std::future;

template<typename IO>
class MpcotReg {
public:
	int party, ngpu;
	int item_n, idx_max, m;
	int tree_height, leave_n;
	int tree_n;
	int consist_check_cot_num;
	bool is_malicious;

	PRG prg;
	IO *netio;
	IO *io;
	block Delta_f2k;
	block *consist_check_chi_alpha = nullptr, *consist_check_VW = nullptr;
	
	std::vector<uint32_t> item_pos_recver;
	GaloisFieldPacking pack;

	MpcotReg(int party, int ngpu, int n, int t, int log_bin_sz, IO *io) {
		this->party = party;
		this->ngpu = ngpu;
		this->io = netio = io;
		consist_check_cot_num = 128;

		this->is_malicious = false;

		this->item_n = t;
		this->idx_max = n;
		this->tree_height = log_bin_sz+1;
		this->leave_n = 1<<(this->tree_height-1);
		this->tree_n = this->item_n;
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
	void mpcot(Mat *outputs, OTPre<IO> * ot, block *pre_cot_data) {
		if(party == BOB) consist_check_chi_alpha = new block[item_n];
		// consist_check_VW = new block[item_n];

		// vector<SPCOT_Sender<IO>*> senders;
		// vector<SPCOT_Recver<IO>*> recvers;

		if(party == ALICE) {
			mpcot_init_sender(ot);
			exec_parallel_sender(ot, outputs);
		} else {
			bool *choice = new bool[tree_n * (tree_height-1)];
			mpcot_init_recver(choice, ot);
			exec_parallel_recver(ot, outputs, choice);
		}

		if(is_malicious)
			consistency_check_f2k(pre_cot_data, tree_n);

		// for (auto p : senders) delete p;
		// for (auto p : recvers) delete p;

		if(party == BOB) delete[] consist_check_chi_alpha;
		// delete[] consist_check_VW;
	}

	void mpcot_init_sender(OTPre<IO> *ot) {
		for(int i = 0; i < tree_n; ++i) {
			ot->choices_sender();
		}
		netio->flush();
		ot->reset();
	}

	void mpcot_init_recver(bool *choice, OTPre<IO> *ot) {
		for(int t = 0; t < tree_n; ++t) {
			ot->choices_recver(choice);
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

	void exec_parallel_sender(OTPre<IO> *ot, Mat *outputs) {
		vector<future<void>> fut;
		
		block *m0 = new block[tree_n*(tree_height-1)];
		block *m1 = new block[tree_n*(tree_height-1)];
		block *secret_sum = new block[tree_n];

		Log::start(Sender, SeedExp);
		cuda_mpcot_sender(outputs, (blk*)m0, (blk*)m1,
			(blk*)secret_sum, tree_n, tree_height-1, (blk*)&Delta_f2k, ngpu);
		Log::end(Sender, SeedExp);
		Log::start(Sender, BaseOT);

		for (int t = 0; t < tree_n; t++) {
			ot->send(m0+t*(tree_height-1), m1+t*(tree_height-1), tree_height-1, io, t);
		}
		io->send_data(secret_sum, tree_n * sizeof(block));
		Log::end(Sender, BaseOT);
		delete[] m0;
		delete[] m1;
	}

	void exec_parallel_recver(OTPre<IO> *ot, Mat *outputs, bool *choice) {
		vector<future<void>> fut;

		block *mc = new block[tree_n*(tree_height-1)];
		block *secret_sum = new block[tree_n];

		Log::start(Recver, BaseOT);
		for (int t = 0; t < tree_n; t++) {
			ot->recv(mc+t*(tree_height-1), choice+t*(tree_height-1), tree_height-1, io, t);
		}
		io->recv_data(secret_sum, tree_n * sizeof(block));
		Log::end(Recver, BaseOT);
		Log::start(Recver, SeedExp);
		cuda_mpcot_recver(outputs, (blk*)mc, (blk*)secret_sum,
			tree_n, tree_height-1, choice, ngpu);
		Log::end(Recver, SeedExp);
		delete[] mc;
	}

	void exec_f2k_sender(SPCOT_Sender<IO> *sender, OTPre<IO> *ot, 
			block *ggm_tree_mem, IO *io, int i) {
		sender->compute(ggm_tree_mem, Delta_f2k);
		sender->template send_f2k<OTPre<IO>>(ot, io, i);
		io->flush();
		if(is_malicious)
			sender->consistency_check_msg_gen(consist_check_VW+i);
	}

	void exec_f2k_recver(SPCOT_Recver<IO> *recver, OTPre<IO> *ot,
			block *ggm_tree_mem, IO *io, int i) {
		recver->template recv_f2k<OTPre<IO>>(ot, io, i);
		recver->compute(ggm_tree_mem);
		if(is_malicious) 
			recver->consistency_check_msg_gen(consist_check_chi_alpha+i, consist_check_VW+i);
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
