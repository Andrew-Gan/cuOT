#include "aesni.h"
#include <pthread.h>

__m128i AES_128_ASSIST (__m128i temp1, __m128i temp2) {
  __m128i temp3;
  temp2 = _mm_shuffle_epi32 (temp2 ,0xff);
  temp3 = _mm_slli_si128 (temp1, 0x4);
  temp1 = _mm_xor_si128 (temp1, temp3);
  temp3 = _mm_slli_si128 (temp3, 0x4);
  temp1 = _mm_xor_si128 (temp1, temp3);
  temp3 = _mm_slli_si128 (temp3, 0x4);
  temp1 = _mm_xor_si128 (temp1, temp3);
  temp1 = _mm_xor_si128 (temp1, temp2);
  return temp1;
}

void aesni_init_ctx(AES_ctx *ctx, const uint8_t *key) {
  __m128i temp1, temp2;
  __m128i *keySchedule = (__m128i*)ctx->roundKey;

  temp1 = _mm_loadu_si128((__m128i*)key);
  keySchedule[0] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1 ,0x1);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[1] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1,0x2);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[2] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1,0x4);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[3] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1,0x8);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[4] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1,0x10);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[5] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1,0x20);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[6] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1,0x40);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[7] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1,0x80);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[8] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1,0x1b);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[9] = temp1;
  temp2 = _mm_aeskeygenassist_si128 (temp1,0x36);
  temp1 = AES_128_ASSIST(temp1, temp2);
  keySchedule[10] = temp1;

  for(int i = 11; i < 20; i++) {
    keySchedule[i] = _mm_aesimc_si128(keySchedule[20 - i]);
  }
}

static void* _encryptWorker(void *args) {
  __m128i tmp;
  int i, j;
  ThreadArgs tArgs = *(ThreadArgs*) args;
  if (tArgs.start >= tArgs.end) {
    return NULL;
  }
  for (i = tArgs.start; i < tArgs.end; i++) {
    tmp = _mm_loadu_si128 (&((__m128i*)tArgs.buf->content)[i]);
    tmp = _mm_xor_si128 (tmp,((__m128i*)tArgs.ctx->roundKey)[0]);
    for(j = 1; j < NUM_ROUNDS; j++){
      tmp = _mm_aesenc_si128 (tmp,((__m128i*)tArgs.ctx->roundKey)[j]);
    }
    tmp = _mm_aesenclast_si128 (tmp,((__m128i*)tArgs.ctx->roundKey)[j]);
    _mm_storeu_si128 (&((__m128i*)tArgs.buf->content)[i],tmp);
  }
  return NULL;
}

void aesni_ecb_encrypt(AES_ctx *ctx, AES_buffer *buf, int numThread) {
  size_t workload = buf->length / AES_BLOCKLEN / numThread;
  int rem = (buf->length / AES_BLOCKLEN) % numThread;
  pthread_t *t = (pthread_t*) malloc((numThread + 1) * sizeof(*t));
  ThreadArgs *args = (ThreadArgs*) malloc((numThread + 1) * sizeof(*args));

  for(int c = 0; c < numThread; c++) {
    args[c] = (ThreadArgs){
      .ctx = ctx, .buf = buf, .start = c * workload, .end = (c + 1) * workload,
    };
    pthread_create(&t[c], NULL, _encryptWorker, &args[c]);
  }
  // handle remaining
  if (rem > 0) {
    args[numThread] = (ThreadArgs){
      .ctx = ctx, .buf = buf,
      .start = numThread * workload,
      .end = numThread * workload + rem,
    };
    pthread_create(&t[numThread], NULL, _encryptWorker, &args[numThread]);
    pthread_join(t[numThread], NULL);
  }

  for(int c = 0; c < numThread; c++) {
    pthread_join(t[c], NULL);
  }

  free(t);
  free(args);
}

static void* _decryptWorker(void *args) {
  __m128i tmp;
  int i, j;
  ThreadArgs tArgs = *(ThreadArgs*)args;
  if (tArgs.start >= tArgs.end) {
    return NULL;
  }
  for (i = tArgs.start; i < tArgs.end; i++) {
    tmp = _mm_loadu_si128 (&((__m128i*)tArgs.buf->content)[i]);
    tmp = _mm_xor_si128 (tmp,((__m128i*)tArgs.ctx->roundKey)[NUM_ROUNDS]);
    for(j = 1; j < NUM_ROUNDS; j++) {
      tmp = _mm_aesdec_si128 (tmp,((__m128i*)tArgs.ctx->roundKey)[NUM_ROUNDS + j]);
    }
    tmp = _mm_aesdeclast_si128 (tmp,((__m128i*)tArgs.ctx->roundKey)[0]);
    _mm_storeu_si128 (&((__m128i*)tArgs.buf->content)[i],tmp);
  }
  return NULL;
}

void aesni_ecb_decrypt(AES_ctx *ctx, AES_buffer *buf, int numThread) {
  size_t workload = buf->length / AES_BLOCKLEN / numThread;
  int rem = (buf->length / AES_BLOCKLEN) % numThread;
  pthread_t *t = (pthread_t*) malloc((numThread + 1) * sizeof(*t));
  ThreadArgs *args = (ThreadArgs*) malloc((numThread + 1) * sizeof(*args));
  for(int c = 0; c < numThread; c++) {
    args[c] = (ThreadArgs){
      .ctx = ctx, .buf = buf, .start = c * workload, .end = (c + 1) * workload,
    };
    pthread_create(&t[c], NULL, _decryptWorker, &args[c]);
  }
  // handle remaining
  args[numThread] = (ThreadArgs){
    .ctx = ctx, .buf = buf,
    .start = numThread * workload,
    .end = numThread * workload + rem,
  };
  pthread_create(&t[numThread], NULL, _decryptWorker, &args[numThread]);

  for(int c = 0; c <= numThread; c++) {
    pthread_join(t[c], NULL);
  }

  free(t);
  free(args);
}
