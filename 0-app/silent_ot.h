#include <curand_kernel.h>
#include <vector>
#include "gpu_block.h"

class SilentOT {
public:
  enum Role { Sender, Recver };
  SilentOT(Role myrole, int myid, int logOT, int numTrees, uint64_t *mychoices);
  void sendBaseOTs();
  void recvBaseOTs();
  std::pair<GPUBlock, GPUBlock> send();
  std::pair<GPUBlock, GPUBlock> recv();

  // placeholders
  void send(GPUBlock &m1, GPUBlock &m2) {}
  GPUBlock recv(uint8_t choice) { return GPUBlock(); }

private:
  Role role;
  size_t id, depth, nTree, numOT;
  curandGenerator_t prng;
  Matrix randMatrix;
  SilentOT *other = nullptr;
  // sender only
  std::vector<std::vector<GPUBlock>> leftHash;
  std::vector<std::vector<GPUBlock>> rightHash;
  // recver only
  uint64_t *choices;
  std::vector<std::vector<GPUBlock>> choiceHash;
};

extern std::array<std::atomic<SilentOT*>, 100> silentOTSenders;
extern std::array<std::atomic<SilentOT*>, 100> silentOTRecvers;
