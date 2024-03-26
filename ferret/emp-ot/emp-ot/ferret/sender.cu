#include "roles.h"

FerretOTSender::FerretOTSender(FerretConfig config) : mConfig(config) {
  mRole = Sender;
  mDev = mConfig.id;
  one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
  base_cot = new BaseCot<T>(mRole, io, malicious);
  cuda_init(mRole);
  ot_pre_data.resize({mConfig.n_pre});

  if(mConfig.runSetup) {
    PRG prg;
    block delta;
    prg.random_block(&delta);
    delta &= one;
    delta ^= 0x1;
    this->Delta = delta;
    cudaMemcpy(ch[1], Delta, sizeof(blk), cudaMemcpyHostToDevice);
    setup(mConfig.preFile);
  }
}

FerretOTSender::~FerretOTSender() {
  cudaSetDevice(mDev);
  block *tmp = new block[mConfig.n_pre];
  cudaMemcpy(tmp, ot_pre_data.data(), mConfig.n_pre*sizeof(block), D2H);
  if (ot_pre_data.size() > 0) {
    write_pre_data128_to_file((void*)tmp, (__uint128_t)Delta, pre_ot_filename);
  }
  delete[] tmp;
  delete base_cot;
  if(lpn != nullptr) delete lpn;
  if(mpcot != nullptr) delete mpcot;
}

void FerretOTSender::extend(Span *ot_output, MpcotReg<T> *mpcot, OTPre<T> *preot,
  Lpn *lpn, Mat &ot_input) {

  cudaSetDevice(mDev);
  send_preot();
  seed_expand(ot_output, mpcot, preot, ot_input);
  primal_lpn(ot_output, mpcot->consist_check_cot_num, lpn, preot, ot_input);
}

void FerretOTSender::setup(std::string pre_file) {
  cudaSetDevice(mDev);
  pre_ot_filename = pre_file != "" ? pre_file : PRE_OT_DATA_REG_SEND_FILE;

  std::future<void> initWorker = std::async([this](){
    cudaSetDevice(mDev);
    extend_initialization();
  });
  cuda_init(mRole);

  bool hasfile = file_exists(pre_ot_filename), hasfile2;
  io->send_data(&hasfile, sizeof(bool));
  io->flush();
  io->recv_data(&hasfile2, sizeof(bool));
  if(hasfile & hasfile2) {
    ch[1] = read_pre_data_from_file(ot_pre_data.data(), pre_ot_filename);
  } else {
    base_cot->cot_gen_pre(Delta);
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

void FerretOTSender::send_preot() {
  cudaSetDevice(mDev);
  Mat tmp({mConfig.n_pre});
  cudaMemcpy(tmp, ot_pre_data.data(), M*sizeof(blk), cudaMemcpyDeviceToDevice);
  pre_ot->send_pre(tmp, Delta);
}

void FerretOTSender::seed_expand(Span &ot_output) {
  seed_expand(ot_output, mpcot, pre_ot, ot_pre_data)
}

void FerretOTSender::seed_expand(Span &ot_output, MpcotReg<T> *mpcot, OTPre<T> *preot, Mat &ot_input) {
  cudaSetDevice(mDev);
  mpcot->sender_init(ch[1]);
  mpcot->mpcot(ot_output, preot, ot_input);
}

void FerretOTSender::primal_lpn(Span &ot_output) {
  primal_lpn(ot_output, mpcot->consist_check_cot_num, lpn, pre_ot, ot_pre_data);
}


void FerretOTSender::primal_lpn(Span &ot_output, uint64_t cot_num, Lpn *lpn,
  OTPre<T> *preot, Mat &ot_input) {
  
  cudaSetDevice(mDev);
  Span kSpan(ot_input, {cot_num});
  lpn->encode(ot_output, kSpan);
  cudaMemcpy(ot_pre_data.data(), ot_buffer.data(ot_limit), M*sizeof(blk), cudaMemcpyDeviceToDevice);
}

void FerretOTSender::online_sender(block *data, int64_t length) {
  cudaSetDevice(mDev);
  bool *bo = new bool[length];
  io->recv_bool(bo, length*sizeof(bool));
  for(int64_t i = 0; i < length; ++i) {
    data[i] = data[i] ^ ch[bo[i]];
  }
  delete[] bo;
}

void FerretOTSender::send_cot(block * data, int64_t length) {
  cudaSetDevice(mDev);
  Mat data_d({length});
  cudaMemcpy(data_d.data(), data, length*sizeof(blk), H2D);
  rcot(data_d);
  cudaMemcpy(data, data_d.data(), length*sizeof(blk), D2H);
  online_sender(data, length);
}
