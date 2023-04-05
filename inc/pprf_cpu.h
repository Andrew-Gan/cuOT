#ifndef __PPRF_H__
#define __PPRF_H__

#include "mytypes.h"

void pprf_sender_cpu(TreeNode *root, size_t depth,
void (*initialiser)(AES_ctx*, const uint8_t*),
void (*encryptor)(AES_ctx*, AES_buffer*, int numThread),
int numTrees);

void pprf_recver_cpu(
void (*initialiser)(AES_ctx*, const uint8_t*),
void (*encryptor)(AES_ctx*, AES_buffer*, int numThread),
TreeNode *d_sparseVec, int *nonZeroRows, size_t depth, int numTrees);

#endif
