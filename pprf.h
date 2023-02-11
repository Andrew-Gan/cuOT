#ifndef __PPRF_H__
#define __PPRF_H__

#include "mytypes.h"

void test_expand(AES_block *root, uint64_t depth, void (*initialiser)(), void (*encryptor)(), size_t nThread, const char *msg);

#endif
