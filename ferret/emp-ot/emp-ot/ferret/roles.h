#ifndef EMP_FERRET_COT_H_
#define EMP_FERRET_COT_H_

#include "mpcot_reg.h"
#include "base_cot.h"
#include "lpnf2.h"
#include "pprf.h"
#include <iostream>
#include <fstream>

#define NGPU 2

static std::string PRE_OT_DATA_REG_SEND_FILE = "./data/pre_ot_data_reg_send";
static std::string PRE_OT_DATA_REG_RECV_FILE = "./data/pre_ot_data_reg_recv";

class PrimalLPNParameter {
public:
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
  const PrimalLPNParameter *lpnParam;
  PprfType pprf;
  uint32_t leftKey[4];
  uint32_t rightKey[4];
  LPNType primalLPN;
  uint64_t *choices;
  std::string *preFile;
  bool runSetup;
  blk delta;
  bool malicious;
};

template<typename T>
class FerretOT : public COT<T>  {
public:
  using COT<T>::io;
  Role mRole;
  int mDev;
  FerretConfig mConfig;
  uint64_t depth, numLeaves;
  uint64_t ot_used, ot_limit;
  MpcotReg<T> *mpcot;
  LpnF2<T> *lpn;
  Mat ot_data;
  OTPre<T> *pre_ot;
  uint64_t pt = 0;
  uint64_t last_round_ot = 0;
  int64_t round_inplace = 0;
  bool round_memcpy = false;

protected:
  blk *ch;
  block one;
  BaseCot<T> *base_cot = nullptr;
  std::string pre_ot_filename;
  bool extend_initialized = false;
  bool is_malicious = false;
  Mat ot_pre_data;
  int64_t M;

public:
  FerretOT(FerretConfig config) : mConfig(config) {
    depth = mConfig.logOT - std::log2(mConfig.lpnParam->t) + 0;
    numLeaves = pow(2, depth);
    malloc_dev((void**)&ch, 2*sizeof(blk));
    memset_dev(&ch[0], 0, sizeof(blk));
  }

  virtual ~FerretOT() { delete io; }
  virtual void seed_expand(Span &ot_output) = 0;

  virtual void rcot_init(Mat &data) {
    std::cout << 0 << std::endl;
    this->io->sync();
    std::cout << 1 << std::endl;
    uint64_t num = data.size();
    std::cout << 2 << std::endl;
    if(ot_data.size() == 0) {
      ot_data.resize({(uint64_t)mConfig.lpnParam->n});
      ot_data.clear();
    }
    std::cout << 3 << std::endl;
    if(extend_initialized == false)
      std::runtime_error("FerretOT::rcot run setup before extending");
    if(num <= silent_ot_left()) {
      memcpy_D2D_dev(data.data(), ot_data.data({ot_used}), num*sizeof(blk));
      ot_used += num;
      return;
    }
    std::cout << 4 << std::endl;
    int64_t gened = silent_ot_left();
    if(gened > 0) {
      memcpy_D2D_dev(data.data(), ot_data.data({ot_used}), gened*sizeof(blk));
      pt += gened;
    }
    std::cout << 5 << std::endl;
    round_inplace = (num-gened-M) / ot_limit;
    last_round_ot = num-gened-round_inplace*ot_limit;
    round_memcpy = last_round_ot>ot_limit?true:false;
    if(round_memcpy) last_round_ot -= ot_limit;

    printf("num OT requested = %ld\n", num);
    printf("OT per iteration = %ld\n", ot_limit);
    printf("iterations = %ld\n", round_inplace);
  }

  virtual void primal_lpn(Span &ot_output) {
    primal_lpn(ot_output, mpcot->consist_check_cot_num, lpn, ot_pre_data);
    memcpy_D2D_dev(ot_pre_data.data(), ot_output.data({ot_limit}), M*sizeof(blk));
    this->ot_used = 0;
  }

protected:
  void write_pre_data_to_file(void* loc, __uint128_t delta, std::string filename) {
    std::ofstream fio(filename, std::ios_base::binary | std::ios_base::out);
    if(!fio.is_open()) {
      std::runtime_error(
        "FerretOT::write_pre_data create directory to store pre-OT data\n"
      );
    }
    else {
      fio << mRole;
      if(mRole == Sender) fio.write((const char*)&delta, sizeof(delta));
      fio << mConfig.lpnParam->n << mConfig.lpnParam->t << mConfig.lpnParam->k;
      fio.write((const char*)loc, mConfig.lpnParam->n_pre*16);
    }
  }

  __uint128_t read_pre_data_from_file(void* pre_loc, std::string filename) {
    std::ifstream fio(filename, std::ios_base::binary | std::ios_base::in);
    uint64_t role;
    fio >> role;
    if(role != mRole)
      std::runtime_error("FerretOT::read_pre_data_from_file wrong mRole");
    __uint128_t delta = 0;
    if(mRole == Sender) fio.read((char*)&delta, sizeof(delta));
    int64_t nin, tin, kin;
    fio >> nin;
    fio >> tin;
    fio >> kin;
    if(nin != mConfig.lpnParam->n || tin != mConfig.lpnParam->t || kin != mConfig.lpnParam->k)
      std::runtime_error("FerretOT::read_pre_data_from_file wrong parameters");
    fio.read((char*)pre_loc, mConfig.lpnParam->n_pre*16);
    std::remove(filename.c_str());
    return delta;
  }

  void extend_initialization() {
    lpn = new LpnF2<T>(mRole, mConfig.lpnParam->n, mConfig.lpnParam->k, io);
    mpcot = new MpcotReg<T>(mRole, mConfig.lpnParam->n, mConfig.lpnParam->t, mConfig.lpnParam->log_bin_sz, io);
    if(is_malicious) mpcot->set_malicious();

    pre_ot = new OTPre<T>(io, mpcot->tree_n*(mpcot->tree_height-1), 1);
    M = mConfig.lpnParam->k + pre_ot->n + mpcot->consist_check_cot_num;
    ot_limit = mConfig.lpnParam->n - M;
    ot_used = ot_limit;
    extend_initialized = true;
  }

  virtual void primal_lpn(Span &ot_output, int cot_num, LpnF2<T> *lpn, Mat &ot_input) {
    set_dev(mDev);
    Span kSpan(ot_input, {(uint64_t)cot_num});
    lpn->encode(ot_output, kSpan);
  }

private:
  uint64_t silent_ot_left() { return ot_limit - ot_used; }
};

#endif
