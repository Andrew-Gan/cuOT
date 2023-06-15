#include <curand.h>
#include <atomic>
#include <vector>
#include "gpu_block.h"
#include "ot.h"
#include "util.h"
#include "aes.h"

class SimplestOT : public OT {
private:
  uint64_t g = 2;
  uint64_t A = 0;
  std::vector<uint64_t> B;
  SimplestOT *other = nullptr;
  size_t n = 0;

  uint8_t buffer[2][320];
  std::array<std::atomic<bool>, 2> hasContent;

  void fromOwnBuffer(uint8_t *d, int id, size_t nBytes);
  void toOtherBuffer(uint8_t *s, int id, size_t nBytes);

public:
  SimplestOT(Role role, int id);
  virtual ~SimplestOT();
  void send(std::vector<GPUBlock> &m0, std::vector<GPUBlock> &m1);
  std::vector<GPUBlock> recv(uint64_t c);
};
