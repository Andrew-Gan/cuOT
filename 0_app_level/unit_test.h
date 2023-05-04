#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

#include "util.h"

void test_rsa();
void test_aes();
void test_cot(Vector d_fullVec, Vector d_puncVec, Vector d_choiceVec, uint8_t delta);

#endif
