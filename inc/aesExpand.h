#ifndef _AESEXPAND_KERNEL_H_
#define _AESEXPAND_KERNEL_H_

#include "util.h"

__global__
void aesExpand128(unsigned *aesKey, TreeNode *leaves,
	unsigned *inData, int expandDir, size_t width);

#endif
