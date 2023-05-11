#include <random>
#include <array>

#include "base_ot.h"

std::array<std::atomic<InitStatus>, 100> initStatuses = {noInit};
std::array<std::atomic<OTStatus>, 100> otStatuses = {notReady};
std::array<std::atomic<BaseOT*>, 100> senders = {nullptr};
std::array<std::atomic<BaseOT*>, 100> recvers = {nullptr};

void BaseOT::sender_init(int id) {
  if (senders[id] != nullptr) {
    fprintf(stderr, "BaseOT sender id still in use: %d\n", id);
    return;
  }
  EventLog::start(BaseOTSenderInit);
  senders[id] = this;
  while (!recvers[id]);
  other = recvers[id];
  rsa = new Rsa();
  other->e = rsa->e;
  other->n = rsa->n;
  initStatuses[id] = rsaInitDone;
  x[0].set(rand());
  x[1].set(rand());
  other->x[0] = x[0];
  other->x[1] = x[1];
  initStatuses[id] = xInitDone;
  EventLog::end(BaseOTSenderInit);
}

void BaseOT::recver_init(int id) {
  if (recvers[id] != nullptr) {
    fprintf(stderr, "BaseOT recver id still in use: %d\n", id);
    return;
  }
  EventLog::start(BaseOTRecverInit);
  recvers[id] = this;
  while (!senders[id]);
  other = senders[id];
  while (initStatuses[id] < rsaInitDone);
  rsa = new Rsa(e, n);
  while (initStatuses[id] < xInitDone);
  EventLog::end(BaseOTRecverInit);
}

BaseOT::BaseOT(Role myrole, int myid): role(myrole), id(myid) {
  if (role == Sender)
    sender_init(id);
  else
    recver_init(id);
}

BaseOT::~BaseOT() {
  delete rsa;
  if (role == Sender) {
    initStatuses[id] = noInit;
    otStatuses[id] = notReady;
    senders[id] = nullptr;
  }
  else
    recvers[id] = nullptr;
}

void BaseOT::send(GPUBlock &m0, GPUBlock &m1) {
  if (role != Sender) {
    fprintf(stderr, "BaseOT not initialised as sender\n");
    return;
  }
  EventLog::start(BaseOTSend);
  while(otStatuses[id] < vReady);
  k[0] = v ^ x[0];
  rsa->decrypt(k[0]);
  k[1] = v ^ x[1];
  rsa->decrypt(k[1]);
  other->mp[0] = m0 ^ k[0];
  other->mp[1] = m1 ^ k[1];
  otStatuses[id] = mReady;
  EventLog::end(BaseOTSend);
}

GPUBlock BaseOT::recv(uint8_t b) {
  if (role != Recver) {
    fprintf(stderr, "BaseOT not initialised as receiver\n");
    return GPUBlock();
  }
  EventLog::start(BaseOTRecv);
  GPUBlock k;
  k.set(rand());
  GPUBlock ke = k;
  rsa->encrypt(ke);
  other->v = x[b] ^ ke;
  otStatuses[id] = vReady;
  while(otStatuses[id] < mReady);
  GPUBlock mb = mp[b] ^ k;
  otStatuses[id] = notReady;
  EventLog::end(BaseOTRecv);
  return mb;
}
