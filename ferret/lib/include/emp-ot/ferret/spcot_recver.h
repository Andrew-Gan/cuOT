#ifndef SPCOT_RECVER_H__
#define SPCOT_RECVER_H__
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "emp-ot/emp-ot.h"
#include "emp-ot/ferret/twokeyprp.h"

#include "cuda_layer.h"

using namespace emp;

template<typename IO>
class SPCOT_Recver {
public:
	vec *ggm_tree;
	mat cSum;
	bool *b;
	int choice_pos, depth, leave_n;
	IO *io;

	blk secret_sum_f2;
	int tree_n;

	SPCOT_Recver(IO *io, int tree_n, int depth_in) {
		this->io = io;
		this->tree_n = tree_n;
		this->depth = depth_in;
		this->leave_n = 1<<(depth_in-1);
		// m = new block[depth];
		cSum.resize(depth, tree_n);
		b = new bool[tree_n*depth];
	}

	~SPCOT_Recver(){
		// delete[] m;
		delete[] b;
	}

	int get_index(int t) {
		choice_pos = 0;
		for(int i = t * depth; i < (t+1) * depth; ++i) {
			choice_pos<<=1;
			if(!b[i])
				choice_pos +=1;
		}
		return choice_pos;
	}

	// receive the message and reconstruct the tree
	// j: position of the secret, begins from 0
	void compute(vec &tree) {
		this->ggm_tree = &tree;
		// ggm_tree_reconstruction(b, m);
		// ggm_tree[choice_pos] = zero_block;
		// block nodes_sum = zero_block;
		// block one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
		// for(int i = 0; i < leave_n; ++i) {
		// 	ggm_tree[i] = ggm_tree[i] & one;
		// 	nodes_sum = nodes_sum ^ ggm_tree[i];
		// }
		// ggm_tree[choice_pos] = nodes_sum ^ secret_sum_f2;

		cuda_spcot_recver_compute(tree_n, leave_n, depth, tree, b, cSum);

		// TBD: confirm handled by code above
		// cudaMemset(ggm_tree.data(choice_pos), 0, sizeof(blk));
		// blk one = { .data = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE} };
		// blk *one_d;
		// cudaMalloc(&one_d, sizeof(*one_d));
		// cudaMemcpy(one_d, &one, sizeof(*one_d), cudaMemcpyHostToDevice);

		// ggm_tree.and_scalar(one_d);
		// vec nodes_sum(leave_n + 1);
		// nodes_sum = ggm_tree;
		// nodes_sum.set(leave_n, secret_sum);
		// nodes_sum.sum(1, leave_n+1);
		// ggm_tree.set(choice_pos, nodes_sum.data(0));
	}

	// receive the message and reconstruct the tree
	// j: position of the secret, begins from 0
	template<typename OT>
	void recv_f2k(OT * ot, IO * io2) {
		block *cSum_cpu = new block[tree_n*depth];

		ot->recv(cSum_cpu, b, tree_n*depth, io2, 0);
		io2->recv_data(&secret_sum_f2, sizeof(blk));

		cuda_memcpy(cSum.data(), cSum_cpu, tree_n*(depth)*sizeof(blk), H2D);

		delete[] cSum_cpu;
	}

	void consistency_check_msg_gen(block *chi_alpha, block *W) {
		// X
		// block *chi = new block[leave_n];
		// Hash hash;
		// block digest[2];
		// hash.hash_once(digest, &secret_sum_f2, sizeof(block));
		// uni_hash_coeff_gen(chi, digest[0], leave_n);
		// *chi_alpha = chi[choice_pos];
		// vector_inn_prdt_sum_red(W, chi, ggm_tree, leave_n);
		// delete[] chi;
	}

	// void ggm_tree_reconstruction(bool *b, block *m) {
	// 	int to_fill_idx = 0;
	// 	TwoKeyPRP prp(zero_block, makeBlock(0, 1));
	// 	for(int i = 1; i < depth; ++i) {
	// 		to_fill_idx = to_fill_idx * 2;
	// 		ggm_tree[to_fill_idx] = ggm_tree[to_fill_idx+1] = zero_block;
	// 		if(b[i-1] == false) {
	// 			layer_recover(i, 0, to_fill_idx, m[i-1], &prp);
	// 			to_fill_idx += 1;
	// 		} else layer_recover(i, 1, to_fill_idx+1, m[i-1], &prp);
	// 	}
	// }

	// void layer_recover(int depth, int lr, int to_fill_idx, block sum, TwoKeyPRP *prp) {
	// 	int layer_start = 0;
	// 	int item_n = 1<<depth;
	// 	block nodes_sum = zero_block;
	// 	int lr_start = lr==0?layer_start:(layer_start+1);

	// 	for(int i = lr_start; i < item_n; i+=2)
	// 		nodes_sum = nodes_sum ^ ggm_tree[i];
	// 	ggm_tree[to_fill_idx] = nodes_sum ^ sum;
	// 	if(depth == this->depth) return;
	// 	if(item_n == 2)
	// 		prp->node_expand_2to4(&ggm_tree[0], &ggm_tree[0]);
	// 	else {
	// 		for(int i = item_n-4; i >= 0; i-=4)
	// 			prp->node_expand_4to8(&ggm_tree[i*2], &ggm_tree[i]);
	// 	}
	// }
};
#endif
