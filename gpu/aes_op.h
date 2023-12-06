#ifndef __AESENCRYPT_H__
#define __AESENCRYPT_H__

#include "gpu_utils.h"

__global__
void aesEncrypt128(uint32_t *key, uint32_t *data);

__global__
void aesDecrypt128(uint32_t *key, uint32_t *data);

__global__
void aesExpand128(uint32_t *keyLeft, uint32_t *keyRight, blk *interleaved,
	blk *separated, uint64_t width);

#endif
