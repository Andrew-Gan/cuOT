#ifndef __AES_H__
#define __AES_H__

#include "util.h"

void aes_init_ctx(AES_ctx* ctx, const uint8_t* key);

#endif
