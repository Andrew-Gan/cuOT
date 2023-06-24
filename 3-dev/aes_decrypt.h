#ifndef __AESDECRYPT_H__
#define __AESDECRYPT_H__

__global__
void aesDecrypt128(unsigned *key, unsigned *result, unsigned *inData);

#endif
