#ifndef __OT_H__
#define __OT_H__

#include <atomic>
#include <vector>
#include "gpu_block.h"

class OT {
protected:
  Role role;
  int id;
  OT *other;

public:
  OT(Role myrole, int myid);
  virtual void send(std::vector<GPUBlock> &m0, std::vector<GPUBlock> &m1) = 0;
  virtual std::vector<GPUBlock> recv(uint64_t b) = 0;
};

extern std::array<std::atomic<OT*>, 100> senders;
extern std::array<std::atomic<OT*>, 100> recvers;

#endif
