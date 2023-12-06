#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <vector>
#include <array>
#include <atomic>
#include "gpu_vector.h"
#include "gpu_matrix.h"
#include "base_ot.h"
#include "expand.h"
#include "compress.h"

#define CHUNK_SIDE (1<<18)

class SilentOTSender;
class SilentOTRecver;

struct SilentOTConfig {
  int id, logOT, nTree;
  BaseOTType baseOT;
  ExpandType expander;
  CompressType compressor;
  uint64_t *choices;
};

class SilentOT {
public:
  SilentOT(SilentOTConfig config) : mConfig(config) {
    depth = mConfig.logOT - log2((float) mConfig.nTree) + 1;
    numOT = pow(2, mConfig.logOT);
    numLeaves = pow(2, depth);
  }
  virtual void run() = 0;

protected:
  SilentOTConfig mConfig;
  uint64_t depth, numOT, numLeaves;
  virtual void base_ot() = 0;
  virtual void pprf_expand() = 0;
  virtual void mult_compress() = 0;
};

class SilentOTSender : public SilentOT {
public:
  Vec fullVector;
  blk *delta = nullptr;
  std::vector<cudaEvent_t> expandEvents;

  SilentOTSender(SilentOTConfig config);
  void run();

private:
  SilentOTRecver *other = nullptr;

  std::vector<Vec> leftHash;
  std::vector<Vec> rightHash;
  virtual void base_ot();

  virtual void pprf_expand();
  virtual void mult_compress();
};

class SilentOTRecver : public SilentOT {
public:
  // receive from sender
  std::vector<Vec> leftBuffer;
  std::vector<Vec> rightBuffer;

  std::atomic<bool> eventsRecorded = false;

  Vec puncVector, choiceVector;

  SilentOTRecver(SilentOTConfig config);
  void run();

private:
  SilentOTSender *other = nullptr;

  std::vector<Vec> choiceHash;
  virtual void base_ot();

  virtual void pprf_expand();
  virtual void mult_compress();
  virtual void get_choice_vector();
};

extern std::array<std::atomic<SilentOTSender*>, 100> silentOTSenders;
extern std::array<std::atomic<SilentOTRecver*>, 100> silentOTRecvers;

#endif
