#ifndef __AESNI_H__
#define __AESNI_H__

#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "mytypes.h"

void aesni_init_ctx(AES_ctx *ctx, const uint8_t *key);

void aesni_ecb_encrypt(AES_ctx *ctx, AES_buffer *buf, int nThreads);
void aesni_ecb_decrypt(AES_ctx *ctx, AES_buffer *buf, int nThreads);

#endif
