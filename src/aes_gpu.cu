#include "aes.h"
#include "aesEncrypt_kernel.h"
#include "aesDecrypt_kernel.h"
#include "aesCudaUtils.hpp"

__host__
cudaTextureObject_t alloc_key_texture(AES_ctx *ctx, cudaResourceDesc *resDesc, cudaTextureDesc *texDesc) {
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
void dealloc_key_texture(cudaTextureObject_t key) {
  cudaResourceDesc resDesc;
  cudaGetTextureObjectResourceDesc(&resDesc, key);
  cudaFree(resDesc.res.linear.devPtr);
  cudaDestroyTextureObject(key);
}

__host__
void gpu_padding(AES_buffer *buf) {
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

void aesgpu_ecb_encrypt(AES_ctx *ctx, AES_buffer *buf, int numThread) {
  size_t len_old = buf->length;
  gpu_padding(buf);
  aesgpu_ecb_xcrypt(ctx, buf, true);
  buf->length = len_old;
}

void aesgpu_ecb_decrypt(AES_ctx *ctx, AES_buffer *buf, int numThread) {
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
