#ifndef __AESENCRYPT_H__
#define __AESENCRYPT_H__

#include "gpu_define.h"

#define T_TABLE_SIZE 256
#define AES_BLOCK_SIZE 16
#define AES_NUM_ROUNDS 10
#define AES_RK_SIZE (AES_BLOCK_SIZE * (AES_NUM_ROUNDS+1))
#define GPU_SHARED_MEM_BANK 32

__global__
void aesEncrypt128(uint32_t* rk, uint32_t* data);

__global__
void aesDecrypt128(uint32_t *key, uint32_t *data);

__global__
void aesExpand128(uint32_t *keyLeft, uint32_t *keyRight, blk *interleaved_in,
	blk *interleaved_out, blk *separated, uint64_t width);

#endif
