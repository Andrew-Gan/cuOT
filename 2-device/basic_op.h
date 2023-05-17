#ifndef __BASIC_OP_H__
#define __BASIC_OP_H__

#include "util.h"

__global__
void xor_gpu(uint8_t *c, uint8_t *a, uint8_t *b, size_t n);

__global__
void xor_circular(uint8_t *c, uint8_t *a, uint8_t *b, size_t len_b, size_t n);

__global__
void and_gpu(Vector c, Vector a, uint8_t b);

__global__
void modular_exp_gpu(uint32_t *b, uint32_t e, uint32_t n);

__global__
void chinese_rem_theorem_gpu(uint32_t *c, uint32_t d, uint32_t p,
  uint32_t q, uint32_t d_p, uint32_t d_q, uint32_t q_inv);

#endif
