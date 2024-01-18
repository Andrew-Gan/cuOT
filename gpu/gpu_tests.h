#ifndef __GPU_TESTS_H__
#define __GPU_TESTS_H__

#include "gpu_vector.h"

void check_cuda();
void check_alloc(void *ptr);
void check_call(const char* msg);
bool check_rot(Vec &m0, Vec &m1, Vec &mc, uint64_t c);
bool check_cot(Vec &full, Vec &punc, Vec &choice, blk *delta);

#endif
