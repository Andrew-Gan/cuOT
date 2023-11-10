#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

#include "util.h"
#include "silent_ot.h"

void test_cuda();
void test_cot(SilentOTSender &sender, SilentOTRecver &recver);
void test_reduce();

#endif
