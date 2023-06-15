#include "simplest_ot.h"
#include "Blake2.h"

using RandomOracle = Blake2;

SimplestOT::SimplestOT(Role role, int id) : OT(role, id) {
  if (role == Sender) {
    while(recvers[id] == nullptr);
    OT *recv = recvers[id];
    other = dynamic_cast<SimplestOT*>(recv);
  }
  else {
    while(senders[id] == nullptr);
    OT *send = senders[id];
    other = dynamic_cast<SimplestOT*>(send);
  }
  hasContent[0] = false;
  hasContent[1] = false;
}

SimplestOT::~SimplestOT() {
  if (role == Sender)
    senders[id] = nullptr;
  else
    recvers[id] = nullptr;
}

void SimplestOT::fromOwnBuffer(uint8_t *d, int id, size_t nBytes) {
  while (!hasContent[id]);
  memcpy(d, buffer[id], nBytes);
  hasContent[id] = false;
}

void SimplestOT::toOtherBuffer(uint8_t *s, int id, size_t nBytes) {
  while (other->hasContent[id]);
  memcpy(other->buffer[id], s, nBytes);
  other->hasContent[id] = true;
}

void SimplestOT::send(std::vector<GPUBlock> &m0, std::vector<GPUBlock> &m1) {
  uint64_t a = rand() & ((1 << 5) - 1);
  A = pow(g, a);
  n = m0.size();
  EventLog::start(BaseOTSend);
  toOtherBuffer((uint8_t*) &A, 0, sizeof(A));
  toOtherBuffer((uint8_t*) &n, 1, sizeof(n));

  A = A * a;
  B.resize(n);
  fromOwnBuffer((uint8_t*) &B.at(0), 0, sizeof(B.at(0)) * B.size());

  for (size_t i = 0; i < n; i++) {
    B.at(i) *= a;
    RandomOracle ro(TREENODE_SIZE);
    ro.Update(B.at(i));
    ro.Update(i);
    uint8_t buff[TREENODE_SIZE];
    cudaMemcpy(buff, m0.at(i).data_d, TREENODE_SIZE, cudaMemcpyDeviceToHost);
    ro.Final(buff);

    B.at(i) -= A;
    ro.Reset();
    ro.Update(B.at(i));
    ro.Update(i);
    cudaMemcpy(buff, m1.at(i).data_d, TREENODE_SIZE, cudaMemcpyDeviceToHost);
    ro.Final(buff);
  }
  EventLog::end(BaseOTSend);
}

std::vector<GPUBlock> SimplestOT::recv(uint64_t c) {
  fromOwnBuffer((uint8_t*) &A, 0, sizeof(A));
  fromOwnBuffer((uint8_t*) &n, 1, sizeof(n));
  EventLog::start(BaseOTRecv);
  std::vector<GPUBlock> res(n);
  std::vector<uint64_t> b(n);
  for (size_t i = 0; i < n; i++) {
    b.at(i) = rand() & ((1 << 5) - 1);
    uint8_t choice = c & (1 << i) >> i;
    uint64_t B0 = pow(g, b.at(i));
    uint64_t B1 = A + B0;
    B.push_back(choice == 0 ? B0 : B1);
  }
  toOtherBuffer((uint8_t*) &B.at(0), 0, sizeof(B.at(0)) * B.size());
  uint8_t buff[TREENODE_SIZE];
  for (size_t i = 0; i < n; i++) {
    uint64_t mB = A * b.at(i);
    RandomOracle ro(TREENODE_SIZE);
    ro.Update(mB);
    ro.Update(i);
    ro.Final(buff);
    cudaMemcpy(res.at(i).data_d, buff, TREENODE_SIZE, cudaMemcpyHostToDevice);
  }
  EventLog::end(BaseOTRecv);
  return res;
}
