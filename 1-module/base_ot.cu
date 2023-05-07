#include <random>
#include <array>

#include "base_ot.h"

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
  initStatus = rsaInitDone;
  other->initStatus = rsaInitDone;
  x[0].set(rand());
  other->x[0] = x[0];
  x[1].set(rand());
  other->x[1] = x[1];
  while(initStatus < aesInitDone);
  rsa->decrypt((uint32_t*) aesKey_enc, 16);
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
  while (initStatus < rsaInitDone);
  rsa = new Rsa(e, n);
  aes = new Aes();
  memcpy(aesKey_enc, aes->key, AES_BLOCKLEN);
  rsa->encrypt((uint32_t*) aesKey_enc, 16);
  memcpy(other->aesKey_enc, aesKey_enc, sizeof(aesKey_enc));
  other->initStatus = initStatus = aesInitDone;
}

BaseOT::BaseOT(Role myrole, int id):
  role(myrole), initStatus(noInit), otStatus(notReady) {
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
}

void BaseOT::send(AesBlocks &m0, AesBlocks &m1) {
  if (role != Sender) {
    fprintf(stderr, "BaseOT not initialised as sender\n");
    return;
  }
  while(otStatus < vReady);
  k0 = v ^ x[0];
  // print_gpu<<<1, 1>>>(k0.data_d, k0.nBlock * 16);
  cudaDeviceSynchronize();
  aes->decrypt(k0);
  k1 = v ^ x[1];
  aes->decrypt(k1);
  other->mp[0] = m0 ^ k0;
  // print_gpu<<<1, 1>>>(other->mp[0].data_d, other->mp[0].nBlock * 16);
  cudaDeviceSynchronize();
  other->mp[1] = m1 ^ k1;
  cudaDeviceSynchronize();
  otStatus = mReady;
  other->otStatus = mReady;
}

AesBlocks BaseOT::recv(uint8_t b) {
  if (role != Recver) {
    fprintf(stderr, "BaseOT not initialised as receiver\n");
    return AesBlocks();
  }
  AesBlocks k;
  k.set(rand());
  AesBlocks k_enc = k;
  // print_gpu<<<1, 1>>>(k_enc.data_d, k_enc.nBlock * 16);
  cudaDeviceSynchronize();
  aes->encrypt(k_enc);
  // print_gpu<<<1, 1>>>(k_enc.data_d, k_enc.nBlock * 16);
  cudaDeviceSynchronize();
  other->v = x[b] ^ k_enc;
  // print_gpu<<<1, 1>>>(other->v.data_d, other->v.nBlock * 16);
  cudaDeviceSynchronize();
  other->otStatus = otStatus = vReady;
  while(otStatus < mReady);
  AesBlocks mb = mp[b] ^ k;
  // print_gpu<<<1, 1>>>(mb.data_d, mb.nBlock * 16);
  cudaDeviceSynchronize();
  otStatus = notReady;
  other->otStatus = notReady;
  return mb;
}
