#ifndef __AES_H__
#define __AES_H__

#include "mytypes.h"

void aes_init_ctx(AES_ctx* ctx, const uint8_t* key);
void aes_ecb_encrypt(AES_ctx* ctx, AES_buffer* buf);
void aes_ecb_decrypt(AES_ctx* ctx, AES_buffer* buf);

#endif
