#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

#include "util.h"

void test_cuda();
// void test_rsa();
void test_aes();
void test_cot(Vector fullVec_d, Vector puncVec_d, Vector choiceVec_d, uint8_t delta);
void test_base_ot();

#endif
