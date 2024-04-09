#ifndef __GPU_TESTS_H__
#define __GPU_TESTS_H__

#include "gpu_matrix.h"

int check_cuda();
void check_alloc(void *ptr);
void check_free_mem();
bool check_rot(Mat &m0, Mat &m1, Mat &mc, uint64_t c);
bool check_cot(Mat &full, Mat &punc, Mat &choice, blk *delta);

#endif
