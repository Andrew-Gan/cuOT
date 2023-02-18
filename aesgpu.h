#ifndef __AESGPU_H__
#define __AESGPU_H__

#include "mytypes.h"

void aesgpu_ecb_encrypt(AES_ctx *ctx, AES_buffer *buf);
void aesgpu_ecb_decrypt(AES_ctx *ctx, AES_buffer *buf);
void aesgpu_tree_expand(TreeNode *tree, size_t depth);

#endif
