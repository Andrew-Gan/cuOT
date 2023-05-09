#include <random>
#include <array>

#include "base_ot.h"

std::array<std::atomic<InitStatus>, 100> initStatuses = {noInit};
std::array<std::atomic<OTStatus>, 100> otStatuses = {notReady};
std::array<std::atomic<BaseOT*>, 100> senders = {nullptr};
std::array<std::atomic<BaseOT*>, 100> recvers = {nullptr};

void BaseOT::sender_init(int id) {
  if (senders[id] != nullptr) {
    fprintf(stderr, "More than one OT sender for tree id: %d\n", id);
    return;
  }
  senders[id] = this;
  while (!recvers[id]);
  other = recvers[id];

  rsa = new Rsa();
  auto [e, n] = rsa->getPublicKey();
  other->e = e;
  other->n = n;
  initStatuses[id] = rsaInitDone;
  x[0].set(0);
  other->x[0] = x[0];
  x[1].set(0);
  other->x[1] = x[1];
  initStatuses[id] = xInitDone;
  while(initStatuses[id] < aesInitDone);
  // rsa->decrypt((uint32_t*) aesKey_enc, 16);
  aes = new Aes(aesKey_enc);
}

void BaseOT::recver_init(int id) {
  if (recvers[id] != nullptr) {
    fprintf(stderr, "More than one OT recver for tree id: %d\n", id);
    return;
  }
  recvers[id] = this;
  while (!senders[id]);
  other = senders[id];
  while (initStatuses[id] < rsaInitDone);
  rsa = new Rsa(e, n);
  while (initStatuses[id] < xInitDone);
  aes = new Aes();
  // memcpy(aesKey_enc, aes->key, AES_BLOCKLEN);
  // rsa->encrypt((uint32_t*) aesKey_enc, 16);
  // memcpy(other->aesKey_enc, aesKey_enc, sizeof(aesKey_enc));
  memcpy(other->aesKey_enc, aes->key, AES_BLOCKLEN);
  initStatuses[id] = aesInitDone;
}

BaseOT::BaseOT(Role myrole, int myid): role(myrole), id(myid) {
  if (role == Sender) {
    sender_init(id);
  }
  else {
    recver_init(id);
  }
}

BaseOT::~BaseOT() {
  delete rsa;
  delete aes;
  if (role == Sender) {
    initStatuses[id] = noInit;
    otStatuses[id] = notReady;
    senders[id] = nullptr;
  }
  else
    recvers[id] = nullptr;
}

void BaseOT::send(AesBlocks &m0, AesBlocks &m1) {
  if (role != Sender) {
    fprintf(stderr, "BaseOT not initialised as sender\n");
    return;
  }
  while(otStatuses[id] < vReady);
  k0 = v ^ x[0];
  aes->decrypt(k0);
  k1 = v ^ x[1];
  aes->decrypt(k1);
  other->mp[0] = m0 ^ k0;
  other->mp[1] = m1 ^ k1;
  otStatuses[id] = mReady;
}

AesBlocks BaseOT::recv(uint8_t b) {
  if (role != Recver) {
    fprintf(stderr, "BaseOT not initialised as receiver\n");
    return AesBlocks();
  }
  AesBlocks k;
  k.set(0);
  AesBlocks k_enc = k;
  aes->encrypt(k_enc);
  other->v = x[b] ^ k_enc;
  otStatuses[id] = vReady;
  while(otStatuses[id] < mReady);
  AesBlocks mb = mp[b] ^ k;
  otStatuses[id] = notReady;
  return mb;
}
