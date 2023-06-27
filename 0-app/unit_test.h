#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

#include "util.h"
#include "gpu_block.h"

void test_cuda();
// void test_rsa();
void test_aes();
void test_cot(GPUBlock &fullVector, GPUBlock &puncVector, GPUBlock &choiceVector, GPUBlock &delta);
void test_base_ot();
void test_reduce();

#endif
