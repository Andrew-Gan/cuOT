#include <curand.h>
#include <atomic>
#include <vector>
#include "gpu_block.h"
#include "ot.h"
#include "util.h"
#include "aes.h"

class SimplestOT : public OT {
private:
  curandGenerator_t prng;
  uint8_t g = 2;
  std::atomic<uint64_t> A = 0;
  std::vector<uint64_t> B;
  std::atomic<bool> eReceived = false;
  std::array<std::vector<GPUBlock>, 2> e;
  SimplestOT *other = nullptr;
  Aes aes0, aes1;
  std::atomic<size_t> count;
  uint8_t* hash(uint64_t v);

public:
  SimplestOT(Role role, int id);
  virtual ~SimplestOT();
  void send(std::vector<GPUBlock> &m0, std::vector<GPUBlock> &m1);
  std::vector<GPUBlock> recv(uint64_t c);
};
