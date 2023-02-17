#ifndef __PPRF_H__
#define __PPRF_H__

#include "mytypes.h"

void aescpu_tree_expand(AES_block *root, uint64_t depth,
void (*initialiser)(AES_ctx*, const uint8_t*),
void (*encryptor)(AES_ctx*, AES_buffer*),
const char *msg);

#endif
