#ifndef __AESGPU_H__
#define __AESGPU_H__

#include "mytypes.h"

extern void aesgpu_ecb_encrypt(AES_ctx *ctx, AES_buffer *buf, int nBlocks);
extern void aesgpu_ecb_decrypt(AES_ctx *ctx, AES_buffer *buf, int nBlocks);

#endif
