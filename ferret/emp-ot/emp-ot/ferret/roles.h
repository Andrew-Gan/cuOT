#ifndef EMP_FERRET_COT_H_
#define EMP_FERRET_COT_H_

#include "emp-ot/ferret/mpcot_reg.h"
#include "emp-ot/ferret/base_cot.h"
#include "emp-ot/ferret/lpn_f2.h"
#include "emp-ot/ferret/constants.h"

#include "pprf.h"
#include "lpn.h"
#include <iostream>
#include <fstream>

#include "cuda_layer.h"
#include "logger.h"

// number of gpu used per party
#define NGPU 2

class FerretOTSender;
class FerretOTRecver;

class PrimalLPNParameter { public:
	int64_t n, t, k, log_bin_sz, n_pre, t_pre, k_pre, log_bin_sz_pre;
	PrimalLPNParameter() {}
	PrimalLPNParameter(int64_t n, int64_t t, int64_t k, int64_t log_bin_sz, int64_t n_pre, int64_t t_pre, int64_t k_pre, int64_t log_bin_sz_pre)
		: n(n), t(t), k(k), log_bin_sz(log_bin_sz),
		n_pre(n_pre), t_pre(t_pre), k_pre(k_pre), log_bin_sz_pre(log_bin_sz_pre) {

		if(n != t * (1<<log_bin_sz) ||
			n_pre != t_pre * (1<< log_bin_sz_pre) ||
			n_pre < k + t * log_bin_sz + 128 )
			error("LPN parameter not matched");	
	}
	int64_t buf_sz() const {
		return n - t * log_bin_sz - k - 128;
	}
};

const static PrimalLPNParameter ferret_b13 = PrimalLPNParameter(10485760, 1280, 452000, 13, 470016, 918, 32768, 9);
const static PrimalLPNParameter ferret_b12 = PrimalLPNParameter(10268672, 2507, 238000, 12, 268800, 1050, 17384, 8);
const static PrimalLPNParameter ferret_b11 = PrimalLPNParameter(10180608, 4971, 124000, 11, 178944, 699, 17384, 8);

struct FerretConfig {
  int id, logOT;
  PrimalLPNParameter lpnParam;
  PprfType pprf;
  uint32_t leftKey[4];
  uint32_t rightKey[4];
  LPNType primalLPN;
  uint64_t *choices;
  std::string preFile;
  bool runSetup = false;
  blk delta;
};

class FerretOT {
public:
  Role mRole;
	FerretConfig mConfig;
  uint64_t depth, numLeaves;
	uint64_t otUsed, otLimit;
  Pprf *mpcot;
	Lpn *lpn;

	FerretOT(FerretConfig config) : mConfig(config) {
    depth = mConfig.logOT - std::log2(mConfig.nTree) + 0;
    numLeaves = pow(2, depth);
  }
	virtual ~FerretOT() {}

  virtual void ot_pre_initialisation() = 0;
  virtual void seed_expand() = 0;
  virtual void primal_lpn() = 0;

private:
	blk ch[2];
	int64_t M;
	bool extend_initialized = false;

	blk one;
	Mat ot_data;
  Mat *buffer;

	Mat ot_pre_data;
	std::string pre_ot_filename;

	BaseCot<T> *base_cot = nullptr;
	OTPre<T> *pre_ot = nullptr;

	virtual void extend_initialization();
	uint64_t silent_ot_left() { return otLimit - otUsed; }

	virtual void write_pre_data_to_file(void* loc, blk *delta, std::string filename) {
    std::ofstream fio(filename, std::ios_base::binary | std::ios_base::out);
    if(!outfile.is_open()) {
      std::cerr << "Attempted to create directory at " << filename << std::endl;
      std::runtime_error(
        "FerretOT::write_pre_data_to_file create directory to store pre-OT data\n"
      );
    }
    else {
      fio << mRole;
      if(mRole == Sender) fio << *delta;
      fio << mConfig.n << mConfig.t << mConfig.k;
      fio.write(loc, mConfig.n_pre*16);
    }
  }

	blk read_pre_data_from_file(void* pre_loc, std::string filename) {
    std::ifstream fio(filename, std::ios_base::binary | std::ios_base::in);
    uint64_t role;
    fio >> role;
    if(role != mRole)
      std::runtime_error("FerretOT::read_pre_data_from_file wrong party");
    blk delta;
    if(mRole == Sender) fio >> delta;
    int64_t nin, tin, kin;
    fio >> nin;
    fio >> tin;int64_t
    fio >> kin;
    if(nin != mConfig.n || tin != mConfig.t || kin != mConfig.k)
      std::runtime_error("FerretOT::read_pre_data_from_file wrong parameters");
    fio.read(pre_loc, mConfig.n_pre*16);
    std::remove(filename.c_str());
    return delta;
  }
};

class FerretOTSender : public FerretOT {
public:
  Mat *fullVector;
  blk *delta;

  FerretOTSender(SilentConfig config) : FerretOT(config) {
    cudaMemcpy(delta, config.delta, sizeof(blk), cudaMemcpyHostToDevice);
    setup(config.preFile);
    cudaMemcpy(ch[1], delta, sizeof(blk), cudaMemcpyDeviceToDevice);
  }
  virtual ~FerretOTSender();
  virtual void ot_pre_initialisation();
  virtual void seed_expand();
  virtual void primal_lpn();
  
  virtual void setup(std::string pre_file);
	virtual void extend(Span &otOutput, OTPre<T> *preot, Mat &ot_input);
	virtual void extend_f2k(Span &otBuffer);
  virtual void send_cot(blk * data, int64_t length) override;
  virtual void online_sender(blk *data, int64_t length);
};

class FerretOTRecver : public FerretOT {
public:
  Mat *puncVector;
  uint64_t *puncPos;
  FerretOTSender *other = nullptr;
  uint64_t *activeParent;

  FerretOTRecver(SilentConfig config) : FerretOT(config) {
    setup(config.preFile);
  }
  virtual ~FerretOTRecver();
  virtual void ot_pre_initialisation();
  virtual void seed_expand();
  virtual void primal_lpn();
  virtual void get_choice_vector();

  virtual void setup(std::string pre_file);
	virtual void extend(Span &otOutput, OTPre<T> *preot, Mat &ot_input);
	virtual void extend_f2k(Span &otBuffer);
	virtual void recv_cot(blk* data, const bool * b, int64_t length) override;
  virtual void online_recver(blk *data, const bool *b, int64_t length);
};

extern std::array<std::atomic<FerretOTSender*>, 8> ferretOTSenders;
extern std::array<std::atomic<FerretOTRecver*>, 8> ferretOTRecvers;

#endif
