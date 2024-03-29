#ifndef __RECVER_H__
#define __RECVER_H__

#include "roles.h"
#include "dev_layer.h"

template<typename T>
class FerretOTRecver : public FerretOT<T> {
public:
  using FerretOT<T>::mRole;
  using FerretOT<T>::mDev;
  using FerretOT<T>::mConfig;
  using FerretOT<T>::ot_limit;
  using FerretOT<T>::mpcot;
  using FerretOT<T>::lpn;
  using FerretOT<T>::pre_ot;
  using FerretOT<T>::pt;
  using FerretOT<T>::ch;
  using FerretOT<T>::one;
  using FerretOT<T>::base_cot;
  using FerretOT<T>::pre_ot_filename;
  using FerretOT<T>::is_malicious;
  using FerretOT<T>::ot_pre_data;
  using FerretOT<T>::M;
  using COT<T>::io;
  using COT<T>::Delta;

  uint64_t *puncPos;
  uint64_t *activeParent;

  FerretOTRecver(FerretConfig config, T *io) : FerretOT<T>(config) {
    mRole = Recver;
    mDev = NGPU - mConfig.id - 1;
    set_dev(mDev);
    this->io = io;
    one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
    base_cot = new BaseCot<T>(mRole, io, mConfig.malicious);
    ot_pre_data.resize({(uint64_t)mConfig.lpnParam->n_pre});

    if(mConfig.runSetup) {
      setup(mConfig.preFile);
    }
  }

  virtual ~FerretOTRecver() {
    set_dev(mDev);
    block *tmp = new block[mConfig.lpnParam->n_pre];
    memcpy_D2H_dev(tmp, ot_pre_data.data(), mConfig.lpnParam->n_pre*sizeof(blk));
    if (ot_pre_data.size() > 0) {
      this->write_pre_data_to_file((void*)tmp, (__uint128_t)0, pre_ot_filename);
    }
    delete[] tmp;
    delete base_cot;
    if(lpn != nullptr) delete lpn;
    if(mpcot != nullptr) delete mpcot;
  }

  virtual void seed_expand(Span &ot_output) {
    this->seed_expand(ot_output, mpcot, pre_ot, ot_pre_data);
  }

  virtual void send_cot(block* data0, int64_t length) {}
  virtual void recv_cot(block* data, const bool* b, int64_t length) {}

private:
  virtual void setup(std::string *pre_file) {
    if(pre_file != nullptr) pre_ot_filename = *pre_file;
    else {
      pre_ot_filename=PRE_OT_DATA_REG_RECV_FILE;
    }

    std::future<void> initWorker = std::async([this](){
      set_dev(mDev);
      this->extend_initialization();
    });

    ot_pre_data.resize({(uint64_t)mConfig.lpnParam->n_pre});
    bool hasfile = file_exists(pre_ot_filename), hasfile2;
    this->io->recv_data(&hasfile2, sizeof(bool));
    this->io->send_data(&hasfile, sizeof(bool));
    this->io->flush();
    if(hasfile & hasfile2) {
      Delta = (block)this->read_pre_data_from_file((void*)ot_pre_data.data(), pre_ot_filename);
    } else {
      base_cot->cot_gen_pre();

      MpcotReg<T> mpcot_ini(mRole, mConfig.lpnParam->n_pre, mConfig.lpnParam->t_pre, mConfig.lpnParam->log_bin_sz_pre, io);
      if(is_malicious) mpcot_ini.set_malicious();
      OTPre<T> pre_ot_ini(io, mpcot_ini.tree_height-1, mpcot_ini.tree_n);
      LpnF2<T> lpn(mRole, mConfig.lpnParam->n_pre, mConfig.lpnParam->k_pre, io);

      block *pre_data_ini = new block[mConfig.lpnParam->k_pre+mpcot_ini.consist_check_cot_num];
      memset_dev(ot_pre_data.data(), 0, mConfig.lpnParam->n_pre*16);

      base_cot->cot_gen(&pre_ot_ini, pre_ot_ini.n);
      base_cot->cot_gen(pre_data_ini, mConfig.lpnParam->k_pre+mpcot_ini.consist_check_cot_num);
      Span pre_d(ot_pre_data);
      Mat pre_data_ini_d({(uint64_t)mConfig.lpnParam->k_pre+mpcot_ini.consist_check_cot_num});
      memcpy_H2D_dev(pre_data_ini_d.data(), pre_data_ini, pre_data_ini_d.size_bytes());
      extend(pre_d, &mpcot_ini, &pre_ot_ini, &lpn, pre_data_ini_d);
      delete[] pre_data_ini;
    }

    initWorker.get();
  }

  virtual void extend(Span &ot_output, MpcotReg<T> *mpcot, OTPre<T> *preot,
    LpnF2<T> *lpn, Mat &ot_input) {
    set_dev(mDev);
    this->seed_expand(ot_output, mpcot, preot, ot_input);
    this->primal_lpn(ot_output, mpcot->consist_check_cot_num, lpn, ot_input);
  }

  virtual void online_recver(blk *data, const bool *b, int64_t length) {
    // set_dev(mDev);
    // bool *bo = new bool[length];
    // for(int64_t i = 0; i < length; ++i) {
    //   bo[i] = b[i] ^ getLSB(data[i]);
    // }
    // this->io->send_bool(bo, length*sizeof(bool));
    // delete[] bo;
  }

  virtual void seed_expand(Span &ot_output, MpcotReg<T> *mpcot, OTPre<T> *preot, Mat &ot_input) {
    set_dev(mDev);
    
    block *tmp = new block[mConfig.lpnParam->n_pre];
    pre_ot->recv_pre(tmp);
    memcpy_H2D_dev(ot_pre_data.data(), tmp, M*sizeof(blk));
    delete[] tmp;

    mpcot->recver_init();
    mpcot->mpcot(ot_output, preot, ot_input);
  }
};

// void FerretOTRecver::recv_cot(block* data, const bool * b, int64_t length) {
//   set_dev(mDev);
//   Mat data_d({length});
//   cudaMemcpy(data_d.data(), data, length*sizeof(blk), cudaMemcpyHostToDevice);
//   rcot(data_d);
//   cudaMemcpy(data, data_d.data(), length*sizeof(blk), cudaMemcpyHDeviceToHost);
//   online_recver(data, b, length);
// }

#endif
