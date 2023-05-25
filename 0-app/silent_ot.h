#include <curand_kernel.h>
#include "ot.h"

class SilentOT : public OT {
private:
  int depth, nTree, numOT;
  curandGenerator_t prng;
  Matrix randMatrix;

public:
  SilentOT(Role myrole, int myid, int logOT, int numTrees);
  void send(GPUBlock &m1, GPUBlock &m2) {}
  GPUBlock recv(uint8_t choice) { return GPUBlock(); }
  std::pair<GPUBlock, GPUBlock> send();
  std::pair<GPUBlock, GPUBlock> recv(uint64_t *choices);
};
