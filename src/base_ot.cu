#include <random>
#include "base_ot.h"

BaseOT *senders[100] = {nullptr};
BaseOT *recvers[100] = {nullptr};

BaseOT::BaseOT(Role role, int id) {
  initStatus = noInit;
  otStatus = notReady;
  if (role == Sender) {
    senderInit(id);
  }
  else {
    recverInit(id);
  }
}

void BaseOT::ot_send(AesBlocks d_m0, AesBlocks d_m1) {
  while(otStatus < vReady);
  d_k0 = v ^ x[0];
  aes.decrypt(&d_k0);
  d_k1 = v ^ x[1];
  aes.decrypt(&d_k1);
  other->d_mp[0] = d_mp[0] = d_m0 ^ d_k0;
  other->d_mp[1] = d_mp[1] = d_m1 ^ d_k1;
  cudaDeviceSynchronize();
  otStatus = mReady;
  other->otStatus = mReady;
}

AesBlocks BaseOT::ot_recv(uint8_t b, size_t nBytes) {
  AesBlocks d_k = rand();
  AesBlocks d_k_enc = d_k;
  aes.encrypt(&d_k_enc);
  other->v = v = x[b] ^ d_k_enc;
  other->otStatus = otStatus = vReady;
  while(otStatus < mReady);
  AesBlocks mb = d_mp[b] ^ d_k;
  otStatus = notReady;
  other->otStatus = notReady;
  return mb;
}

void BaseOT::senderInit(int id) {
  if (senders[id] != nullptr) {
    fprintf(stderr, "More than one OT sender for tree id: %d\n", id);
    return;
  }
  senders[id] = this;
  while (!recvers[id]);
  other = recvers[id];

  auto [e, n] = rsa.getPublicKey();
  other->e = e;
  other->n = n;
  initStatus = rsaInitDone;
  other->initStatus = rsaInitDone;
  other->x[0] = x[0] = rand();
  other->x[1] = x[1] = rand();
  while(initStatus < aesInitDone);
  rsa.decrypt((uint32_t*) aesKey_enc, 16);
  aes = Aes(aesKey_enc);
}

void BaseOT::recverInit(int id) {
  if (recvers[id] != nullptr) {
    fprintf(stderr, "More than one OT recver for tree id: %d\n", id);
    return;
  }
  recvers[id] = this;
  while (!senders[id]);
  other = senders[id];

  while (initStatus < rsaInitDone);
  rsa = Rsa(e, n);
  memcpy(aesKey_enc, aes.key, AES_BLOCKLEN);
  rsa.encrypt((uint32_t*) aesKey_enc, 16);
  memcpy(other->aesKey_enc, aesKey_enc, sizeof(aesKey_enc));
  other->initStatus = initStatus = aesInitDone;
}
