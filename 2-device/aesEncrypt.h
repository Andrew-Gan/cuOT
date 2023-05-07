#ifndef __AESENCRYPT_H__
#define __AESENCRYPT_H__

__global__
void aesEncrypt128(unsigned *key, unsigned * result, unsigned * inData);

#endif
