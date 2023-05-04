#include <random>
#include "base_ot.h"

BaseOT *senders[100] = {nullptr};
BaseOT *recvers[100] = {nullptr};

BaseOT::BaseOT(Role role, int id) {
  initStatus = noInit;
  otStatus = notReady;
  if (role == Sender) {
    sender_init(id);
  }
  else {
    recver_init(id);
  }
}

void BaseOT::send(AesBlocks m0, AesBlocks m1) {
  while(otStatus < vReady);
  k0 = v ^ x[0];
  aes.decrypt(&k0);
  k1 = v ^ x[1];
  aes.decrypt(&k1);
  other->mp[0] = mp[0] = m0 ^ k0;
  other->mp[1] = mp[1] = m1 ^ k1;
  cudaDeviceSynchronize();
  otStatus = mReady;
  other->otStatus = mReady;
}

AesBlocks BaseOT::recv(uint8_t b) {
  AesBlocks k = rand();
  AesBlocks k_enc = k;
  aes.encrypt(&k_enc);
  other->v = v = x[b] ^ k_enc;
  other->otStatus = otStatus = vReady;
  while(otStatus < mReady);
  AesBlocks mb = mp[b] ^ k;
  otStatus = notReady;
  other->otStatus = notReady;
  return mb;
}

void BaseOT::sender_init(int id) {
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

void BaseOT::recver_init(int id) {
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
