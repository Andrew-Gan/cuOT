#ifndef __AESENCRYPT_H__
#define __AESENCRYPT_H__

#include "gpu_define.h"

__global__
void aesEncrypt(uint32_t *rk, uint32_t *data);

#ifdef USE_IMPROVED_AES
__global__
void aesExpand(uint32_t *rkLeft, uint32_t *rkRight, blk *mixed_in,
	blk *mixed_out, blk *separated, uint64_t width);
#else
__global__
void aesExpand(uint32_t *rk, blk *mixed_in, blk *mixed_out,
	blk *separated, uint64_t width, int expandDir);
#endif // USE_IMPROVED_AES

#endif // __AESENCRYPT_H__