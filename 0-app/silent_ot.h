#include <curand_kernel.h>
#include <vector>
#include "gpu_block.h"

class SilentOT {
public:
  enum Role { Sender, Recver };
  SilentOT(Role myrole, int myid, int logOT, int numTrees, uint64_t *mychoices);
  std::pair<GPUBlock, GPUBlock> send();
  std::pair<GPUBlock, GPUBlock> recv();

private:
  Role role;
  size_t id, depth, nTree, numOT;
  curandGenerator_t prng;
  Matrix randMatrix;
  SilentOT *other = nullptr;

  // network
  std::atomic<size_t> msgDelivered = 0;
  std::vector<std::vector<GPUBlock>> leftHash;
  std::vector<std::vector<GPUBlock>> rightHash;

  // sender only
  void baseOT_send();
  std::pair<GPUBlock, GPUBlock> pprf_send(TreeNode root);

  // recver only
  uint64_t *choices;
  std::vector<std::vector<GPUBlock>> choiceHash;
  void baseOT_recv();
  std::pair<GPUBlock, SparseVector> pprf_recv(uint64_t *choices);
};

extern std::array<std::atomic<SilentOT*>, 100> silentOTSenders;
extern std::array<std::atomic<SilentOT*>, 100> silentOTRecvers;
