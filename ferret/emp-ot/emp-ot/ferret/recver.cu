#include "roles.h"

FerretOTRecver::FerretOTRecver(FerretConfig config) : mConfig(config) {
  mRole = Recver;
  mDev = NGPU - mConfig.id - 1;
  cudaSetDevice(mDev);
  one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
  base_cot = new BaseCot<T>(mRole, io, malicious);
  cuda_init(mRole);
  ot_pre_data.resize({mConfig.n_pre});

  if(mConfig.runSetup) {
    setup(mConfig.preFile);
  }
}

FerretOTRecver::~FerretOTRecver() {
  cudaSetDevice(mDev);
  block *tmp = new block[mConfig.n_pre];
  cuda_memcpy(tmp, ot_pre_data.data(), mConfig.n_pre*sizeof(blk), D2H);
  if (ot_pre_data.size() > 0) {
    if(mRole == ALICE) write_pre_data128_to_file((void*)tmp, (__uint128_t)Delta, pre_ot_filename);
    else write_pre_data128_to_file((void*)tmp, (__uint128_t)0, pre_ot_filename);
  }
  delete[] tmp;
  delete base_cot;
  if(lpn != nullptr) delete lpn;
  if(mpcot != nullptr) delete mpcot;
}

// extend f2k in detail
void FerretOTRecver::extend(Span *ot_output, MpcotReg<T> *mpcot, OTPre<T> *preot,
    Lpn<T> *lpn, Mat &ot_input) {
  
  cudaSetDevice(mDev);
  recv_preot();
  seed_expand(ot_output, mpcot, preot, ot_input);
  primal_lpn(ot_output, mpcot->consist_check_cot_num, lpn, preot, ot_input);
}

void FerretOTRecver::setup(std::string pre_file) {
  cudaSetDevice(mDev);
  pre_ot_filename = pre_file != "" ? pre_file : PRE_OT_DATA_REG_RECV_FILE;

  std::future<void> initWorker = std::async([this](){
    cudaSetDevice(mDev);
    extend_initialization();
  });
  cuda_init(mRole);

  bool hasfile = file_exists(pre_ot_filename), hasfile2;
  io->recv_data(&hasfile2, sizeof(bool));
  io->send_data(&hasfile, sizeof(bool));
  io->flush();
  if(hasfile & hasfile2) {
    Delta = read_pre_data_from_file(ot_pre_data.data(), pre_ot_filename);
  } else {
    base_cot->cot_gen_pre();
    MpcotReg<T> mpcot_ini(mRole, mConfig.n_pre, mConfig.t_pre, mConfig.log_bin_sz_pre, ios);
    if(is_malicious) mpcot_ini.set_malicious();
    OTPre<T> pre_ot_ini(ios, mpcot_ini.tree_n*(mpcot_ini.tree_height-1), 1);
    Lpn lpn(mRole, mConfig.n_pre, mConfig.k_pre, io);

    block *pre_data_ini = new block[mConfig.k_pre+mpcot_ini.consist_check_cot_num];
    ot_pre_data.clear();

    base_cot->cot_gen(&pre_ot_ini, pre_ot_ini.n);
    base_cot->cot_gen(pre_data_ini, mConfig.k_pre+mpcot_ini.consist_check_cot_num);
    Mat tmp({mConfig.k_pre+mpcot_ini.consist_check_cot_num});
    cudaMemcpy(tmp.data(), pre_data_ini, tmp.size_bytes(), H2D);
    extend(ot_pre_data, &mpcot_ini, &pre_ot_ini, &lpn, tmp);
    delete[] pre_data_ini;
  }

  initWorker.get();
}

void FerretOTRecver::recv_preot() {
  cudaSetDevice(mDev);
  Mat tmp(mConfig.n_pre);
  pre_ot->recv_pre(tmp);
  cudaMemcpy(ot_pre_data.data(), tmp, M*sizeof(blk), cudaMemcpyDeviceToDevice);
}

void FerretOTRecver::seed_expand(Span &ot_output) {
  seed_expand(ot_output, mpcot, pre_ot, ot_pre_data);
}

void FerretOTRecver::seed_expand(Span &ot_output, MpcotReg<T> *mpcot,
  OTPre<T> *preot, Mat &ot_input) {
  
  cudaSetDevice(mDev);
  mpcot->recver_init();
  mpcot->mpcot(ot_output, preot, ot_input);
}

void FerretOTRecver::primal_lpn(Span &ot_output) {
  primal_lpn(ot_output, mpcot->consist_check_cot_num, lpn, pre_ot, ot_pre_data);
}

void FerretOTRecver::primal_lpn(Span &ot_output, uint64_t cot_num, Lpn *lpn,
  OTPre<T> *preot, Mat &ot_input) {
  
  cudaSetDevice(mDev);
  Span kSpan(ot_input, {cot_num});
  lpn->encode(ot_output, kSpan);
  cudaMemcpy(ot_pre_data.data(), ot_buffer.data(ot_limit), M*sizeof(blk), cudaMemcpyDeviceToDevice);
}

void FerretOTRecver::online_recver(block *data, const bool *b, int64_t length) {
  cudaSetDevice(mDev);
  bool *bo = new bool[length];
  for(int64_t i = 0; i < length; ++i) {
    bo[i] = b[i] ^ getLSB(data[i]);
  }
  io->send_bool(bo, length*sizeof(bool));
  delete[] bo;
}

void FerretOTRecver::recv_cot(block* data, const bool * b, int64_t length) {
  cudaSetDevice(mDev);
  Mat data_d({length});
  cuda_memcpy(data_d.data(), data, length*sizeof(blk), H2D);
  rcot(data_d);
  cuda_memcpy(data, data_d.data(), length*sizeof(blk), D2H);
  online_recver(data, b, length);
}
