#include <random>
#include "base_ot.h"

BaseOT::BaseOT(Role role, size_t msgSize) {
  if (role == Sender) {
    auto [ke, kn] = rsa.getPublicKey();
    BaseOT::e = ke;
    BaseOT::n = kn;
    initStatus = rsaInitDone;
    BaseOT::x[0] = rand();
    BaseOT::x[1] = rand();
    while(initStatus < aesInitDone);
    rsa.decrypt((uint32_t*) aesKey_enc, 16);
    aes = Aes(aesKey_enc);
  }
  else {
    while (initStatus < rsaInitDone);
    rsa = Rsa(BaseOT::e, BaseOT::n);
    cudaMemcpy(aesKey_enc, aes.d_key, AES_BLOCKLEN, cudaMemcpyDeviceToHost);
    rsa.encrypt((uint32_t*) aesKey_enc, 16);
    initStatus = aesInitDone;
  }
}

void BaseOT::ot_send(AesBlocks d_m0, AesBlocks d_m1) {
  while(otStatus < vReady);
  d_k0 = BaseOT::v ^ BaseOT::x[0];
  aes.decrypt(d_k0);
  d_k1 = BaseOT::v ^ BaseOT::x[1];
  aes.decrypt(d_k1);
  d_mp[0] = d_m0 ^ d_k0;
  d_mp[1] = d_m1 ^ d_k1;
  cudaDeviceSynchronize();
  otStatus = mReady;
}

AesBlocks BaseOT::ot_recv(uint8_t b, size_t nBytes) {
  AesBlocks d_k = rand();
  AesBlocks d_k_enc = d_k;
  aes.encrypt(d_k_enc);
  v = x[b] ^ d_k_enc;
  otStatus = vReady;
  while(otStatus < mReady);
  AesBlocks mb = d_mp[b] ^ d_k;
  otStatus = notReady;
  return mb;
}
