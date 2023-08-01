#ifndef _AESEXPAND_KERNEL_H_
#define _AESEXPAND_KERNEL_H_

#include "util.h"

__global__
void aesExpand128(uint32_t *keyLeft, uint32_t *keyRight, OTblock *interleaved,
	OTblock *left, OTblock *right, OTblock *inData, uint64_t width);

#endif
