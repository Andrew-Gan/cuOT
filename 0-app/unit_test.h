#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

#include "util.h"
#include "gpu_vector.h"

void test_cuda();
void test_cot(GPUvector<OTblock> &fullVector, OTblock *delta,
  GPUvector<OTblock> &puncVector, GPUvector<OTblock> &choiceVector);
void test_base_ot();
void test_reduce();

#endif
