#include "ot.h"

class SilentOT : public OT {
private:
  int depth, nTree, numOT;

public:
  SilentOT(Role role, int id, int logOT, int numTrees);
  void send(GPUBlock &m0, GPUBlock &m1);
  GPUBlock recv(uint8_t choice);
  GPUBlock recv(uint64_t *choices);
};
