/***************************************************************************
 *   Copyright (C) 2006                                                    *
 *                                                                         *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

/**
	@author Svetlin Manavski <svetlin@manavski.com>
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "aes.h"
#include "aesgpu.h"

#include "sbox_E.h"
#include "sbox_D.h"
#include "aesEncrypt_kernel.h"
#include "aesDecrypt_kernel.h"
#include "aesExpand_kernel.h"

#include <vector>
#include "aesCudaUtils.hpp"

#define PADDED_LEN 1024

__host__
static void cuda_check() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
    fprintf(stderr, "There is no device.\n");
  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (deviceProp.major >= 1)
      break;
  }
  if (dev == deviceCount)
    fprintf(stderr, "There is no device supporting CUDA.\n");
  else
    cudaSetDevice(dev);
}

__host__
static cudaTextureObject_t alloc_key_texture(AES_ctx *ctx, cudaResourceDesc *resDesc, cudaTextureDesc *texDesc) {
  unsigned *d_Key;
  cudaMalloc(&d_Key, AES_keyExpSize);
  cudaMemcpy(d_Key, ctx->roundKey, AES_keyExpSize, cudaMemcpyHostToDevice);
  memset(resDesc, 0, sizeof(*resDesc));
  resDesc->resType = cudaResourceTypeLinear;
  resDesc->res.linear.devPtr = d_Key;
  resDesc->res.linear.desc.f = cudaChannelFormatKindUnsigned;
  resDesc->res.linear.desc.x = 32;
  resDesc->res.linear.sizeInBytes = AES_keyExpSize;

  memset(texDesc, 0, sizeof(*texDesc));
  texDesc->readMode = cudaReadModeElementType;

  cudaTextureObject_t texKey;
  cudaCreateTextureObject(&texKey, resDesc, texDesc, NULL);

  return texKey;
}

__host__
static void dealloc_key_texture(cudaTextureObject_t key) {
  cudaResourceDesc resDesc;
  cudaGetTextureObjectResourceDesc(&resDesc, key);
  cudaFree(resDesc.res.linear.devPtr);
  cudaDestroyTextureObject(key);
}

__host__
static void gpu_padding(AES_buffer *buf) {
  unsigned mod16 = buf->length % 16;
  unsigned div16 = buf->length / 16;

  unsigned padElem = 16 - mod16;

  for (unsigned cnt = 0; cnt < padElem; ++cnt)
    buf->content[div16*16 + mod16 + cnt] = padElem;

  buf->length += padElem;
  buf->length += PADDED_LEN - (buf->length % PADDED_LEN);
}

__host__
static void aesgpu_ecb_xcrypt(AES_ctx *ctx, AES_buffer *buf, bool isEncrypt) {
  cuda_check();

  unsigned *d_Input, *d_Result;
  cudaMalloc(&d_Input, buf->length);
  cudaMalloc(&d_Result, buf->length);
  cudaMemset(d_Result, 0, buf->length);

  cudaResourceDesc resDesc;
  cudaTextureDesc texDesc;
  cudaTextureObject_t texKey = alloc_key_texture(ctx, &resDesc, &texDesc);

	dim3 threads(BSIZE, 1);
  dim3 grid(buf->length / BSIZE / 4, 1);
  cudaMemcpy(d_Input, buf->content, buf->length, cudaMemcpyHostToDevice);

  if (isEncrypt)
	  aesEncrypt128<<<grid, threads>>>(texKey, d_Result, d_Input);
  else
    aesDecrypt128<<<grid, threads>>>(texKey, d_Result, d_Input);

  cudaDeviceSynchronize();
  cudaMemcpy(buf->content, d_Result, buf->length, cudaMemcpyDeviceToHost);

  cudaFree(d_Input);
  cudaFree(d_Result);
  dealloc_key_texture(texKey);
}

void aesgpu_ecb_encrypt(AES_ctx *ctx, AES_buffer *buf) {
  size_t len_old = buf->length;
  gpu_padding(buf);
  aesgpu_ecb_xcrypt(ctx, buf, true);
  buf->length = len_old;
}

void aesgpu_ecb_decrypt(AES_ctx *ctx, AES_buffer *buf) {
  // invert expanded key
  std::vector<unsigned> key(AES_KEYLEN);
  for(int i = 0; i < AES_KEYLEN; i++) {
    key.at(i) = ctx->roundKey[i];
  }
  std::vector<unsigned> expKey(AES_keyExpSize);
  expFunc(key, expKey);

  std::vector<unsigned> invExpKey(AES_keyExpSize);
  invExpFunc(expKey, invExpKey);
  for(int i = 0; i < AES_keyExpSize; i++) {
    ctx->roundKey[i] = invExpKey.at(i);
  }

  size_t len_old = buf->length;
  gpu_padding(buf);
  aesgpu_ecb_xcrypt(ctx, buf, false);
  buf->length = len_old;
}

void aesgpu_tree_expand(AES_block *tree, size_t depth) {
  cuda_check();
  cudaFree(0);
  size_t maxWidth = pow(2, depth);
  size_t numNode = maxWidth * 2 - 1;

  // keys to use for tree expansion
  AES_ctx aesKeys[2];
  uint64_t k0 = 3242342;
  uint8_t k0_blk[16] = {0};
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  aes_init_ctx(&aesKeys[0], k0_blk);

  uint64_t k1 = 8993849;
  uint8_t k1_blk[16] = {0};
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  aes_init_ctx(&aesKeys[1], k1_blk);

  cudaResourceDesc resDescLeft;
  cudaResourceDesc resDescRight;
  cudaTextureDesc texDesc;

  // store key in texture memory
  cudaTextureObject_t texLKey = alloc_key_texture(&aesKeys[0], &resDescLeft, &texDesc);
  cudaTextureObject_t texRKey = alloc_key_texture(&aesKeys[1], &resDescRight, &texDesc);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int i = 0; i < NUM_SAMPLES; i++) {
    // store tree in device memory
    AES_block *d_Tree;
    // double size as threads will write directly
    cudaMalloc(&d_Tree, sizeof(*tree) * numNode);
    cudaMemcpy(d_Tree, tree, sizeof(*tree), cudaMemcpyHostToDevice);

    AES_block *d_InputBuf;
    cudaMalloc(&d_InputBuf, sizeof(*d_InputBuf) * maxWidth / 2 + PADDED_LEN);

    size_t layerStartIdx = 1, width = 2;
    cudaStream_t *s = (cudaStream_t*) malloc(sizeof(*s) * depth);
    for (size_t d = 1 ; d <= depth; d++, width *= 2) {
      cudaStreamCreate(&s[d-1]);
      // copy previous layer for expansion
      cudaMemcpy(d_InputBuf, &d_Tree[(layerStartIdx - 1) / 2], sizeof(*d_Tree) * width / 2, cudaMemcpyDeviceToDevice);

      size_t paddedLen = (width / 2) * AES_BLOCKLEN;
      paddedLen += 16 - (paddedLen % 16);
      paddedLen += PADDED_LEN - (paddedLen % PADDED_LEN);
      static int thread_per_aesblock = 4;
      dim3 grid(paddedLen * thread_per_aesblock / 16 / BSIZE, 1);
      dim3 thread(BSIZE, 1);
      aesExpand128<<<grid, thread>>>(texLKey, d_Tree,  (unsigned*) d_InputBuf, layerStartIdx, width);
      aesExpand128<<<grid, thread>>>(texRKey, d_Tree,  (unsigned*) d_InputBuf, layerStartIdx + 1, width);

      cudaDeviceSynchronize();

      cudaMemcpyAsync(&tree[layerStartIdx], &d_Tree[layerStartIdx], sizeof(*tree) * width, cudaMemcpyDeviceToHost, s[d-1]);

      layerStartIdx += width;
    }

    cudaDeviceSynchronize();
    for (size_t d = 1; d <= depth; d++) {
      cudaStreamDestroy(s[d-1]);
    }
    cudaFree(s);
    cudaFree(d_Tree);
    cudaFree(d_InputBuf);
  }

  dealloc_key_texture(texLKey);
  dealloc_key_texture(texRKey);

  clock_gettime(CLOCK_MONOTONIC, &end);

  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Tree expansion using AESGPU: %0.4f ms\n", duration / NUM_SAMPLES);

}
