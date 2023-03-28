#ifndef __AESGPU_H__
#define __AESGPU_H__

#include "mytypes.h"

void aesgpu_ecb_encrypt(AES_ctx *ctx, AES_buffer *buf, int numThread);
void aesgpu_ecb_decrypt(AES_ctx *ctx, AES_buffer *buf, int numThread);
void pprf_sender_gpu(TreeNode *root, TreeNode *leaves, size_t depth);
void pprf_recver_gpu(TreeNode *root, TreeNode *leaves, size_t depth);

#endif
