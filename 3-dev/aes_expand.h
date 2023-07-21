#ifndef _AESEXPAND_KERNEL_H_
#define _AESEXPAND_KERNEL_H_

#include "util.h"
#include "aes.h"

__global__
void aesExpand128(unsigned *aesKey, OTblock *interleaved, uint32_t *separated,
	unsigned *inData, int expandDir, uint64_t width);

#endif
