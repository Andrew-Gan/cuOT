#include "silentOT.h"

std::array<std::atomic<SilentOTSender*>, 100> silentOTSenders;
std::array<std::atomic<SilentOTRecver*>, 100> silentOTRecvers;

SilentOT::SilentOT(int myid, int logOT, int numTrees) {
  id = myid;
  nTree = numTrees;
  depth = logOT - log2((float) nTree) + 1;
  numOT = pow(2, logOT);
}
