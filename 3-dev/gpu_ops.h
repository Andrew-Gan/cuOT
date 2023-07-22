#ifndef __LOGIC_OPS_H__
#define __LOGIC_OPS_H__

#include "util.h"
#include "cufft.h"

__global__
void and_gpu(uint8_t *a, uint8_t *b, uint64_t n);

__global__
void xor_gpu(uint8_t *a, uint8_t *b, uint64_t n);

__global__
void poly_mod_gpu(uint64_t *data, uint64_t terms);

__global__
void and_single_gpu(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n);

__global__
void xor_single_gpu(uint8_t *a, uint8_t *b, uint64_t size, uint64_t n);

__global__
void bit_transposer(uint64_t *out, uint64_t *in);

__global__
void int_to_float(float *o, uint64_t *i);

__global__
void float_to_int(uint64_t *o, float *i);

__global__
void complex_dot_product(cufftComplex *c, cufftComplex *a, cufftComplex *b);

__global__
void xor_reduce_gpu(uint64_t *data);

__global__
void xor_reduce_packer_gpu(uint64_t *data, uint64_t width);

__global__
void print_gpu(void *data, uint64_t n);

__global__
void print_gpu(void *data, uint64_t n, uint64_t stride);

#endif
