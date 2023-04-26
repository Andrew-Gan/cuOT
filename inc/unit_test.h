#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

#include "util.h"

// test A ^ C =  B & delta
bool unit_test_correlation(Vector d_fullVec, Vector d_puncVec, Vector d_choiceVec, uint8_t delta);

#endif
