#ifndef __AESGPU_H__
#define __AESGPU_H__

#include "mytypes.h"

cudaTextureObject_t alloc_key_texture(AES_ctx *ctx, cudaResourceDesc *resDesc, cudaTextureDesc *texDesc);
void dealloc_key_texture(cudaTextureObject_t key);
void gpu_padding(AES_buffer *buf);

void aesgpu_ecb_encrypt(AES_ctx *ctx, AES_buffer *buf, int numThread);
void aesgpu_ecb_decrypt(AES_ctx *ctx, AES_buffer *buf, int numThread);

#endif
