#include <curand.h>
#include <atomic>
#include "gpu_block.h"
#include "ot.h"
#include "util.h"
#include "aes.h"

class SimplestOT : public OT {
private:
  curandGenerator_t prng;
  uint8_t g = 2;
  uint32_t A = 0, B = 0;
  bool eReceived = false;
  std::array<GPUBlock, 2> e = { GPUBlock(16), GPUBlock(16) };
  SimplestOT *other = nullptr;
  Aes *aes0, *aes1;
  uint8_t* hash(uint64_t v);

public:
  SimplestOT(Role role, int id);
  virtual ~SimplestOT();
  void send(GPUBlock &m0, GPUBlock &m1);
  GPUBlock recv(uint8_t c);
};
