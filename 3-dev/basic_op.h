#ifndef __BASIC_OP_H__
#define __BASIC_OP_H__

#include "util.h"

__global__
void xor_gpu(uint8_t *c, uint8_t *a, uint8_t *b, uint64_t n);

__global__
void xor_circular(uint8_t *c, uint8_t *a, uint8_t *b, uint64_t len_b, uint64_t n);

__global__
void and_gpu(uint8_t *c, uint8_t *a, uint64_t n);

__global__
void xor_reduce_gpu(uint64_t *g_data);

#endif
