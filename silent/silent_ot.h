#ifndef __SILENT_OT_H__
#define __SILENT_OT_H__

#include <vector>
#include <array>
#include <atomic>
#include "gpu_matrix.h"
#include "base_ot.h"
#include "pprf.h"
#include "quasi_cyclic.h"

// optimization options
#define USE_CUFFT_FOR_POLYMUL
#define USE_IMPROVED_COMPLEX_PROD
#define USE_TYPE_CONVERT_AND_MOD

class SOTSender;
class SOTRecver;

struct SilentConfig {
  int id, logOT;
  uint64_t nTree;
  BaseOTType baseOT;
  PprfType pprf;
  uint32_t leftKey[4];
  uint32_t rightKey[4];
  LPNType dualLPN;
  uint64_t *choices;
  int gpuPerParty;
};

class SOT {
public:
  Role mRole;
  SilentConfig mConfig;
  uint64_t mDepth, numOT, numLeaves;
  Pprf *expander;
  QuasiCyclic *lpn;
  // base OT
  Mat m0, m1;

  SOT(SilentConfig config) : mConfig(config) {
    mDepth = mConfig.logOT - std::log2(mConfig.nTree) + 0;
    numOT = pow(2, mConfig.logOT);
    numLeaves = pow(2, mDepth);
  }
  virtual ~SOT() {}
  void generate() {
    base_ot();
    seed_exp();
    dual_lpn();
  }
  virtual void base_ot() = 0;
  virtual void seed_exp() = 0;
  virtual void dual_lpn() = 0;

protected:
  // pprf expansion
  Mat *sep;
  Mat *buffer;
  int mGPU;
};

class SOTSender : public SOT {
public:
  Mat *fullVector;
  blk *delta;
  static blk *m0_h, *m1_h;

  SOTSender(SilentConfig config);
  virtual ~SOTSender();
  virtual void base_ot();
  virtual void seed_exp();
  virtual void dual_lpn();
};

class SOTRecver : public SOT {
public:
  Mat *puncVector;
  Mat choiceVector;
  uint64_t *puncPos = nullptr;
  SOTSender *other = nullptr;
  static blk *mc_h;
  Mat mc;
  uint64_t *activeParent;

  SOTRecver(SilentConfig config);
  virtual ~SOTRecver();
  virtual void base_ot();
  virtual void get_punc_key();
  virtual void seed_exp();
  virtual void dual_lpn();

private:
  virtual void get_choice_vector();
};

extern std::array<std::atomic<SOTSender*>, 16> SOTSenders;
extern std::array<std::atomic<SOTRecver*>, 16> SOTRecvers;

#endif
