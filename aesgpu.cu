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
#include "aesgpu.h"

#include "sbox_E.h"
#include "sbox_D.h"
#include "aesEncrypt_kernel.h"
#include "aesDecrypt_kernel.h"

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
static void aesgpu_ecb_xcrypt(AES_ctx *ctx, AES_buffer *buf, bool isEncrypt) {
  cuda_check();

  unsigned *d_Input, *d_Result, *d_Key;
  cudaMalloc((void**) &d_Input, buf->length);
  cudaMalloc((void**) &d_Key, AES_keyExpSize);
  cudaMalloc((void**) &d_Result, buf->length);
  cudaMemset(d_Result, 0, buf->length);

  cudaMemcpy(d_Key, ctx->roundKey, AES_keyExpSize, cudaMemcpyHostToDevice);

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_Key;
  resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
  resDesc.res.linear.desc.x = 32;
  resDesc.res.linear.sizeInBytes = AES_keyExpSize;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaTextureObject_t texKey;
  cudaCreateTextureObject(&texKey, &resDesc, &texDesc, NULL);

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
  cudaFree(d_Key);
  cudaFree(d_Result);
  cudaDestroyTextureObject(texKey);
}

extern "C" void aesgpu_ecb_encrypt(AES_ctx *ctx, AES_buffer *buf) {
  size_t len_old = buf->length;
  // padding for corner jobs
  unsigned mod16 = buf->length % 16;
  unsigned div16 = buf->length / 16;

  unsigned padElem = 16 - mod16;

  for (unsigned cnt = 0; cnt < padElem; ++cnt)
    buf->content[div16*16 + mod16 + cnt] = padElem;

  buf->length += padElem;
  buf->length = buf->length - (buf->length % PADDED_LEN) + PADDED_LEN;
  aesgpu_ecb_xcrypt(ctx, buf, true);
  buf->length = len_old;
}

extern "C" void aesgpu_ecb_decrypt(AES_ctx *ctx, AES_buffer *buf) {
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
  buf->length = buf->length - (buf->length % PADDED_LEN) + PADDED_LEN;
  aesgpu_ecb_xcrypt(ctx, buf, false);
  buf->length = len_old;
}

extern "C" void aesgpu_tree_expand(cudaTextureObject_t *key, AES_block *tree, size_t depth) {
  cudaTextureObject_t leftKey = key[0], rightKey = key[1];
  int maxWidth = pow(2, depth);
  int numNode = maxWidth * 2 - 1;

  AES_block *leftBuffer, *rightBuffer;
  cudaMalloc(&leftBuffer, sizeof(*leftBuffer) * maxWidth);
  cudaMalloc(&rightBuffer, sizeof(*rightBuffer) * maxWidth);

  
}
