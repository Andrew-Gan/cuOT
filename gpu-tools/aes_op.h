#ifndef __AESENCRYPT_H__
#define __AESENCRYPT_H__

#include "gpu_tools.h"

__global__
void aesEncrypt128(uint32_t *key, uint32_t * result, uint32_t *inData);

__global__
void aesDecrypt128(uint32_t *key, uint32_t *result, uint32_t *inData);

__global__
void aesExpand128(uint32_t *keyLeft, uint32_t *keyRight, blk *interleaved,
	blk *separated, blk *inData, uint64_t width);

#endif
