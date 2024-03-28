#ifndef SPCOT_RECVER_H__
#define SPCOT_RECVER_H__
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "emp-ot/emp-ot.h"
#include "emp-ot/ferret/twokeyprp.h"
#include "gpu_span.h"
#include "pprf.h"

#include "dev_layer.h"

template<typename IO>
class SPCOT_Recver {
public:
	Span *ggm_tree;
	Mat cSum;
	bool *b;
	uint64_t choice_pos, depth, leave_n;
	IO *io;

	blk secret_sum_f2;
	uint64_t tree_n;

	SPCOT_Recver(IO *io, uint64_t tree_n, uint64_t depth_in) {
		this->io = io;
		this->tree_n = tree_n;
		this->depth = depth_in;
		this->leave_n = 1<<(depth_in-1);
		// m = new block[depth];
		cSum.resize({(uint64_t)depth-1, (uint64_t)tree_n});
		b = new bool[tree_n*(depth-1)];
	}

	~SPCOT_Recver(){
		delete[] b;
	}

	int get_index(uint64_t t) {
		choice_pos = 0;
		for(uint64_t i = t * (depth-1); i < (t+1) * (depth-1); ++i) {
			choice_pos<<=1;
			if(!b[i])
				choice_pos +=1;
		}
		return choice_pos;
	}

	// receive the message and reconstruct the tree
	// j: position of the secret, begins from 0
	void compute(Span &tree) {
		this->ggm_tree = &tree;
		uint32_t k0_blk[4] = {3242342};
		uint32_t k1_blk[4] = {8993849};
		Mat buffer({tree.size()});
		Span bufferSpan(buffer);
    	Span *input = &bufferSpan;
		Span *output = &tree;
		AesExpand aesExpand((uint8_t*) k0_blk, (uint8_t*) k1_blk);
		Mat separated({tree.size()});
		uint64_t *activeParent;
		bool *choice;

		malloc_dev((void**)&activeParent, tree_n * sizeof(uint64_t));
		malloc_dev((void**)&choice, depth * tree_n);
		memset_dev(activeParent, 0, tree_n * sizeof(uint64_t));
		memcpy_H2D_dev(choice, b, depth * tree_n);
		for (uint64_t d = 0, inWidth = 1; d < depth-1; d++, inWidth *= 2) {
			std::swap(input, output);
			aesExpand.expand(*input, *output, separated, tree_n*inWidth);
			separated.sum(2 * tree_n, inWidth);
			SPCOT_recver_compute_dev(tree_n, cSum, inWidth, activeParent,
				separated, tree, depth, d, choice);
		}
		if (output != &tree) {
			memcpy_D2D_dev(tree.data(), output->data(), tree.size_bytes());
		}
		free_dev(choice);
		free_dev(activeParent);
	}

	// receive the message and reconstruct the tree
	// j: position of the secret, begins from 0
	template<typename OT>
	void recv_f2k(OT * ot, IO * io2) {
		block *cSum_cpu = new block[tree_n*(depth-1)];
		ot->recv(cSum_cpu, b, tree_n*(depth-1), io2, 0);
		io2->recv_data(&secret_sum_f2, sizeof(blk));
		memcpy_H2D_dev(cSum.data(), cSum_cpu, tree_n*(depth-1)*sizeof(blk));
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
};
#endif
