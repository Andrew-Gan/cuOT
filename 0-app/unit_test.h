#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

#include "util.h"
#include "gpu_vector.h"

void test_cuda();
// void test_rsa();
void test_aes();
void test_cot(GPUvector<OTblock> &fullVector, GPUvector<OTblock> &puncVector, GPUvector<OTblock> &choiceVector, GPUdata &delta);
void test_base_ot();
void test_reduce();

#endif
