#include <cmath>
#include "diffie_hellman.h"

SimplestOT::SimplestOT(Role role, int id) {
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 1234ULL);
  if (role == Sender) {
    senders[id] = this;
    while(recvers[id] == nullptr);
    other = recvers[id];
  }
  else {
    recvers[id] = this;
    while(senders[id] == nullptr);
    other = senders[id];
  }
}

SimplestOT::~SimplestOT() {
  curandDestroyGenerator(prng);
  if (role == Sender)
    senders[id] = nullptr;
  else
    recvers[id] = nullptr;
  delete aes;
}

void SimplestOT::send(GPUBlock &m0, GPUBlock &m1) {
  uint8_t a = rand() % 64;
  other->A = pow(g, a);
  while(B == 0);
  uint64_t k0 = hash(pow(B, a));
  uint64_t k1 = hash(pow(B / A), a);
  aes0 = new Aes(k0);
  aes1 = new Aes(k1);
  other->e0 = aes0.encrypt(m0);
  other->e1 = aes1.encrypt(m1);
  other->eReceived = true;
}

GPUBlock SimplestOT::recv(uint8_t c) {
  uint8_t b = rand() % 64;
  while (A == 0);
  B = pow(g, b);
  if (c == 1)
    B *= A;
  other->B = B;
  uint64_t kb = hash(pow(A, b));
  while(other->A == 0;)
  uint64_t k = pow(A, b);
  aes0 = new Aes(k);
  while(!eReceived);
  return aes0.decrypt(e[c]);
}
