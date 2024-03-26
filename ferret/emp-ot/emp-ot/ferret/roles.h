#ifndef EMP_FERRET_COT_H_
#define EMP_FERRET_COT_H_

#include "mpcot_reg.h"
#include "base_cot.h"
#include "lpnf2.h"

#include "pprf.h"
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
  blk *delta;
};

class FerretOT : public COT<NetIO>  {
public:
  using COT<NetIO>::io;
  using COT<NetIO>::Delta;
  Role mRole;
  int mDev;
	FerretConfig mConfig;
  uint64_t depth, numLeaves;
	uint64_t ot_used, ot_limit;
  MpcotReg<NetIO> *mpcot;
	LpnF2<NetIO> *lpn;
	Mat ot_data;

	FerretOT(FerretConfig config) : mConfig(config) {
    depth = mConfig.logOT - std::log2(mConfig.lpnParam.t) + 0;
    numLeaves = pow(2, depth);
    cudaMalloc(&ch, 2*sizeof(blk));
    cudaMemset(&ch[0], 0, sizeof(blk));
  }
	virtual ~FerretOT() {}
  virtual void seed_expand(Span &ot_output) = 0;
  virtual void primal_lpn(Span &ot_output) = 0;

  virtual void rcot_init(Mat &data) {
    int64_t num = (int64_t) data.size();
    if(ot_data.size() == 0) {
      ot_data.resize({(uint64_t)mConfig.lpnParam.n});
      ot_data.clear();
    }
    if(extend_initialized == false)
      std::runtime_error("FerretOT::rcot run setup before extending");
    if(num <= silent_ot_left()) {
      cudaMemcpy(data.data(), ot_data.data({ot_used}), num*sizeof(blk), cudaMemcpyDeviceToDevice);
      ot_used += num;
      return;
    }
    uint64_t pt = 0;
    int64_t gened = silent_ot_left();
    if(gened > 0) {
      cudaMemcpy(data.data(), ot_data.data({ot_used}), gened*sizeof(blk), cudaMemcpyDeviceToDevice);
      pt += gened;
    }
    int64_t round_inplace = (num-gened-M) / ot_limit;
    int64_t last_round_ot = num-gened-round_inplace*ot_limit;
    bool round_memcpy = last_round_ot>ot_limit?true:false;
    if(round_memcpy) last_round_ot -= ot_limit;

    printf("num OT requested = %ld\n", num);
    printf("OT per iteration = %ld\n", ot_limit);
    printf("iterations = %ld\n", round_inplace);
  }

protected:
	blk *ch;

private:
	int64_t M;
	bool extend_initialized = false;
  bool is_malicious = false;

	blk one;
  Mat *buffer;

	Mat ot_pre_data;
	std::string pre_ot_filename;

	BaseCot<NetIO> *base_cot = nullptr;
	OTPre<NetIO> *pre_ot = nullptr;

	void extend_initialization() {
    cuda_init(mRole);
    lpn = new LpnF2(mRole, mConfig.lpnParam.n, mConfig.lpnParam.k, io);
    mpcot = new MpcotReg<NetIO>(mRole, mConfig.lpnParam.n, mConfig.lpnParam.t, mConfig.lpnParam.log_bin_sz, io);
    if(is_malicious) mpcot->set_malicious();

    pre_ot = new OTPre<NetIO>(io, mpcot->tree_n*(mpcot->tree_height-1), 1);
    M = mConfig.lpnParam.k + pre_ot->n + mpcot->consist_check_cot_num;
    ot_limit = mConfig.lpnParam.n - M;
    ot_used = ot_limit;
    extend_initialized = true;
  }

	uint64_t silent_ot_left() { return ot_limit - ot_used; }

	void write_pre_data_to_file(void* loc, blk *delta, std::string filename) {
    std::ofstream fio(filename, std::ios_base::binary | std::ios_base::out);
    if(!fio.is_open()) {
      std::runtime_error(
        "FerretOT::write_pre_data create directory to store pre-OT data\n"
      );
    }
    else {
      fio << mRole;
      blk d;
      cudaMemcpy(&d, delta, sizeof(blk), cudaMemcpyDeviceToHost);
      if(mRole == Sender) fio.write((const char*)&d, sizeof(blk));
      fio << mConfig.lpnParam.n << mConfig.lpnParam.t << mConfig.lpnParam.k;
      fio.write((const char*)loc, mConfig.lpnParam.n_pre*16);
    }
  }

	blk* read_pre_data_from_file(void* pre_loc, std::string filename) {
    std::ifstream fio(filename, std::ios_base::binary | std::ios_base::in);
    uint64_t role;
    fio >> role;
    if(role != mRole)
      std::runtime_error("FerretOT::read_pre_data_from_file wrong party");
    blk d;
    if(mRole == Sender) fio.read((char*)&d, sizeof(blk));
    blk *delta;
    cudaMalloc(&delta, sizeof(blk));
    cudaMemcpy(delta, &d, sizeof(blk), cudaMemcpyDeviceToHost);
    int64_t nin, tin, kin;
    fio >> nin;
    fio >> tin;
    fio >> kin;
    if(nin != mConfig.lpnParam.n || tin != mConfig.lpnParam.t || kin != mConfig.lpnParam.k)
      std::runtime_error("FerretOT::read_pre_data_from_file wrong parameters");
    fio.read((char*)pre_loc, mConfig.lpnParam.n_pre*16);
    std::remove(filename.c_str());
    return delta;
  }
};

class FerretOTSender : public FerretOT {
public:
  Mat *fullVector;

  FerretOTSender(FerretConfig config) : FerretOT(config) {
    cudaMemcpy(delta, config.delta, sizeof(blk), cudaMemcpyHostToDevice);
    setup(config.preFile);
    cudaMemcpy(&ch[1], delta, sizeof(blk), cudaMemcpyDeviceToDevice);
  }
  virtual ~FerretOTSender();
  virtual void seed_expand(Span &ot_output);
  virtual void primal_lpn(Span &ot_output);

private:
  virtual void setup(std::string pre_file);
	virtual void extend(Span &otOutput, OTPre<NetIO> *preot, Mat &ot_input);
	virtual void extend_f2k(Span &otBuffer);
  virtual void send_cot(blk * data, int64_t length);
  virtual void online_sender(blk *data, int64_t length);

  virtual void seed_expand(Span *ot_output, MpcotReg<NetIO> *mpcot,
    OTPre<NetIO> *preot, Mat &ot_input);
};

class FerretOTRecver : public FerretOT {
public:
  Mat *puncVector;
  uint64_t *puncPos;
  FerretOTSender *other = nullptr;
  uint64_t *activeParent;

  FerretOTRecver(FerretConfig config) : FerretOT(config) {
    setup(config.preFile);
  }
  virtual ~FerretOTRecver();
  virtual void seed_expand(Span &ot_output);
  virtual void primal_lpn(Span &ot_output);
  virtual void get_choice_vector();

private:
  virtual void setup(std::string pre_file);
	virtual void extend(Span &otOutput, OTPre<NetIO> *preot, Mat &ot_input);
	virtual void extend_f2k(Span &otBuffer);
	virtual void recv_cot(blk* data, const bool * b, int64_t length);
  virtual void online_recver(blk *data, const bool *b, int64_t length);

  virtual void seed_expand(Span *ot_output, MpcotReg<NetIO> *mpcot,
    OTPre<NetIO> *preot, Mat &ot_input);
};

extern std::array<std::atomic<FerretOTSender*>, 8> ferretOTSenders;
extern std::array<std::atomic<FerretOTRecver*>, 8> ferretOTRecvers;

#endif
