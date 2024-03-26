#ifndef SPCOT_SENDER_H__
#define SPCOT_SENDER_H__
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "emp-ot/emp-ot.h"
#include "emp-ot/ferret/twokeyprp.h"

#include "cuda_layer.h"

using namespace emp;

template<typename IO>
class SPCOT_Sender {
public:
	blk seed;
	blk *delta;
	Span *ggm_tree;
	Mat lSum, rSum;
	IO *io;
	uint64_t depth, leave_n;
	PRG prg;
	blk secret_sum_f2;
	uint64_t tree_n;

	SPCOT_Sender(IO *io, uint64_t tree_n, uint64_t depth_in) {
		this->tree_n = tree_n;
		initialization(io, depth_in);
		block seed128;
		prg.random_block(&seed128, 1);
		memcpy(&seed, &seed128, sizeof(block));
	}

	void initialization(IO *io, uint64_t depth_in) {
		this->io = io;
		this->depth = depth_in;
		this->leave_n = 1<<(this->depth);
		lSum.resize({depth-1, tree_n});
		rSum.resize({depth-1, tree_n});
	}
	
	// generate GGM tree, transfer secret, F2^k
	void compute(Span &tree, blk *secret) {
		this->ggm_tree = &tree;
		this->delta = secret;
		cuda_spcot_sender_compute(tree, tree_n, depth, lSum, rSum);

		// memset(secretSum, 0, sizeof(secretSum));
		// blk one = { .data = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE} };
		// blk *one_d;
		// cudaMalloc(&one_d, sizeof(*one_d));
		// cudaMemcpy(one_d, &one, sizeof(*one_d), cudaMemcpyHostToDevice);

		// ggm_tree.and_scalar(one_d);
		// Mat nodes_sum(leave_n + 1);
		// nodes_sum = ggm_tree;
		// nodes_sum.set(leave_n, secret);
		// nodes_sum.sum(1, leave_n+1);
		// secret_sum = nodes_sum.data(0);
	}

	// send the nodes by oblivious transfer, F2^k
	template<typename OT>
	void send_f2k(OT * ot, IO * io2) {
		block *lSum_cpu = new block[tree_n*(depth-1)];
		block *rSum_cpu = new block[tree_n*(depth-1)];

		cuda_memcpy(lSum_cpu, lSum.data(), tree_n*(depth-1)*sizeof(blk), D2H);
		cuda_memcpy(rSum_cpu, rSum.data(), tree_n*(depth-1)*sizeof(blk), D2H);

		ot->send(lSum_cpu, rSum_cpu, tree_n*(depth-1), io2, 0);
		io2->send_data(&secret_sum_f2, sizeof(blk));

		delete[] lSum_cpu;
		delete[] rSum_cpu;
	}

	void consistency_check_msg_gen(block *V) {
		// X
		// block *chi = new block[leave_n];
		// Hash hash;
		// block digest[2];
		// hash.hash_once(digest, &secret_sum_f2, sizeof(block));
		// uni_hash_coeff_gen(chi, digest[0], leave_n);

		// vector_inn_prdt_sum_red(V, chi, ggm_tree, leave_n);
		// delete[] chi;
	}

	// void ggm_tree_gen(block *ot_msg_0, block *ot_msg_1, block* ggm_tree_mem, block secret) {
	// 	ggm_tree_gen(ot_msg_0, ot_msg_1, ggm_tree_mem);
	// 	secret_sum_f2 = zero_block;
	// 	block one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
	// 	for(int i = 0; i < leave_n; ++i) {
	// 		ggm_tree[i] = ggm_tree[i] & one;
	// 		secret_sum_f2 = secret_sum_f2 ^ ggm_tree[i];
	// 	}
	// 	secret_sum_f2 = secret_sum_f2 ^ secret;
	// }

	// // generate GGM tree from the top
	// void ggm_tree_gen(block *ot_msg_0, block *ot_msg_1, block* ggm_tree_mem) {
	// 	this->ggm_tree = ggm_tree_mem;
	// 	TwoKeyPRP *prp = new TwoKeyPRP(zero_block, makeBlock(0, 1));
	// 	prp->node_expand_1to2(ggm_tree, seed);
	// 	ot_msg_0[0] = ggm_tree[0];
	// 	ot_msg_1[0] = ggm_tree[1];
	// 	prp->node_expand_2to4(&ggm_tree[0], &ggm_tree[0]);
	// 	ot_msg_0[1] = ggm_tree[0] ^ ggm_tree[2];
	// 	ot_msg_1[1] = ggm_tree[1] ^ ggm_tree[3];
	// 	for(int h = 2; h < depth-1; ++h) {
	// 		ot_msg_0[h] = ot_msg_1[h] = zero_block;
	// 		int sz = 1<<h;
	// 		for(int i = sz-4; i >=0; i-=4) {
	// 			prp->node_expand_4to8(&ggm_tree[i*2], &ggm_tree[i]);
	// 			ot_msg_0[h] ^= ggm_tree[i*2];
	// 			ot_msg_0[h] ^= ggm_tree[i*2+2];
	// 			ot_msg_0[h] ^= ggm_tree[i*2+4];
	// 			ot_msg_0[h] ^= ggm_tree[i*2+6];
	// 			ot_msg_1[h] ^= ggm_tree[i*2+1];
	// 			ot_msg_1[h] ^= ggm_tree[i*2+3];
	// 			ot_msg_1[h] ^= ggm_tree[i*2+5];
	// 			ot_msg_1[h] ^= ggm_tree[i*2+7];
	// 		}
	// 	}
	// 	delete prp;
	// }
};

#endif
