#include <curand.h>
#include <atomic>
#include <vector>
#include "gpu_block.h"
#include "util.h"
#include "aes.h"

class SimplestOT {
public:
  enum Role { Sender, Recver };
  SimplestOT(Role role, int id);
  virtual ~SimplestOT();
  std::array<std::vector<GPUBlock>, 2> send(size_t count);
  std::vector<GPUBlock> recv(size_t count, uint64_t choice);

private:
  Role role;
  int id;
  uint64_t g = 2;
  uint64_t A = 0;
  std::vector<uint64_t> B;
  SimplestOT *other = nullptr;
  size_t n = 0;
  uint8_t buffer[2][320];
  std::array<std::atomic<bool>, 2> hasContent;

  void fromOwnBuffer(uint8_t *d, int id, size_t nBytes);
  void toOtherBuffer(uint8_t *s, int id, size_t nBytes);
};

extern std::array<std::atomic<SimplestOT*>, 100> simplestOTSenders;
extern std::array<std::atomic<SimplestOT*>, 100> simplestOTRecvers;
