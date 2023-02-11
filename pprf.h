#ifndef __PPRF_H__
#define __PPRF_H__

#include "mytypes.h"

void test_expand(AES_block *root, uint64_t depth, void (*initialiser)(), void (*encryptor)(), const char *msg);

#endif
