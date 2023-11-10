#ifndef __TREE_EXP_H__
#define __TREE_EXP_H__

#include "gpu_tools.h"

void gpu_ggm_tree_send(vec &leftSum, vec &rightSum,
    blk *ggm_tree, blk& secret_sum, blk& secret, int depth);

void gpu_ggm_tree_recv(blk *ggm_tree, bool *choices,
    vec &sums, blk& secret_sum, uint64_t choice_pos);

#endif
