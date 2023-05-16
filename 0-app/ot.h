#ifndef __OT_H__
#define __OT_H__

#include <atomic>
#include "gpu_block.h"

class OT {
protected:
  Role role;
  int id;
  OT *other;

public:
  OT(Role myrole, int myid);
  virtual void send(GPUBlock &m0, GPUBlock &m1) = 0;
  virtual GPUBlock recv(uint8_t b) = 0;
};

static std::array<std::atomic<OT*>, 100> senders;
static std::array<std::atomic<OT*>, 100> recvers;

#endif
