#ifndef _AESEXPAND_KERNEL_H_
#define _AESEXPAND_KERNEL_H_

#include "util.h"

// 4 thread 1 block variant
__global__
void aesExpand128_4_1(uint32_t *keyLeft, uint32_t *keyRight, OTblock *interleaved,
	OTblock *separated, OTblock *inData, uint64_t width);

// 1 thread 1 block variant
__global__
void aesExpand128_1_1(uint32_t *keyLeft, uint32_t *keyRight, OTblock *interleaved,
    OTblock *separated, OTblock *inData, uint64_t width);

#endif
