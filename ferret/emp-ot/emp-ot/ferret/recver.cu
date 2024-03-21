#include "roles.h"

FerretOTRecver::FerretOTRecver(FerretConfig config) {
	mRole = Recver;
	one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
	ch[0] = zero_block;
	base_cot = new BaseCot<T>(mRole, io, malicious);
	cuda_init(mRole);
	ot_pre_data.resize({mConfig.n_pre});

	if(mConfig.runSetup) {
		setup(mConfig.preFile);
	}
}

FerretOTRecver::~FerretOTRecver() {
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

void FerretOTRecver::extend_initialization() {
	cuda_init(mRole);
	lpn = new LpnF2<T, 10>(mRole, mConfig.n, mConfig.k, io);
	mpcot = new MpcotReg<T>(mRole, mConfig.n, mConfig.t, mConfig.log_bin_sz, ios);
	if(is_malicious) mpcot->set_malicious();

	pre_ot = new OTPre<T>(io, mpcot->tree_n*(mpcot->tree_height-1), 1);
	M = mConfig.k + pre_ot->n + mpcot->consist_check_cot_num;
	ot_limit = mConfig.n - M;
	ot_used = ot_limit;
	extend_initialized = true;
}

// extend f2k in detail
void FerretOTRecver::extend(Span *ot_output, MpcotReg<T> *mpcot, OTPre<T> *preot,
		LpnF2<T, 10> *lpn, Mat &ot_input) {

	blk Delta_blk;
	memcpy(&Delta_blk, &Delta, sizeof(blk));
	if(mRole == ALICE) mpcot->sender_init(Delta_blk);
	else mpcot->recver_init();

	mpcot->mpcot(ot_output, preot, ot_input);

	Log::start(mRole-1, LPN);
	Span kSpan = ot_input.span(mpcot->consist_check_cot_num);
	Log::mem(mRole-1, LPN);
	lpn->compute(ot_output, kSpan);
	Log::mem(mRole-1, LPN);
	Log::end(mRole-1, LPN);
}

// extend f2k (customized location)
void FerretOTRecver::extend_f2k(Span &ot_buffer) {
	Mat tmp(mConfig.n_pre);
  pre_ot->recv_pre(tmp);
  cudaMemcpy(ot_pre_data.data(), tmp, M*sizeof(blk), cudaMemcpyDeviceToDevice);
	extend(ot_buffer, mpcot, pre_ot, lpn, ot_pre_data);
	cudaMemcpy(ot_pre_data.data(), ot_buffer.data(ot_limit), M*sizeof(blk), cudaMemcpyDeviceToDevice);
	ot_used = 0;
}

// extend f2k
void FerretOTRecver::extend_f2k() {
	Span tmp = ot_data.span();
	extend_f2k(tmp);
}

void FerretOTRecver::setup(block Deltain, std::string pre_file) {
	this->Delta = Deltain;
	setup(pre_file);
	ch[1] = Delta;
}

void FerretOTRecver::setup(std::string pre_file) {
  pre_ot_filename = pre_file != "" ? pre_file : PRE_OT_DATA_REG_RECV_FILE;

	std::future<void> initWorker = std::async([this](){
		extend_initialization();
	});
	cuda_init(mRole);

	bool hasfile = file_exists(pre_ot_filename), hasfile2;
  io->recv_data(&hasfile2, sizeof(bool));
  io->send_data(&hasfile, sizeof(bool));
  io->flush();
	if(hasfile & hasfile2) {
		block *buffer = new block[mConfig.n_pre];
		Delta = (block)read_pre_data128_from_file((void*)buffer, pre_ot_filename);
		cudaMemcpy(ot_pre_data.data(), buffer, mConfig.n_pre*sizeof(blk), cudaMemcpyDeviceToDevice);
		delete buffer;
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
		lpn.init();
		extend(ot_pre_data, &mpcot_ini, &pre_ot_ini, &lpn, tmp);
		delete[] pre_data_ini;
	}

	initWorker.get();
}

void FerretOTRecver::rcot(Mat &data) {
	int64_t num = (int64_t) data.size();
	if(ot_data.size() == 0) {
		ot_data.resize({mConfig.n});
		ot_data.clear();
	}
	if(extend_initialized == false)
		std::runtime_error("FerretOTRecver::rcot run setup before extending");
	if(num <= silent_ot_left()) {
		cuda_memcpy(data.data(), ot_data.data(ot_used), num*sizeof(blk), D2D);
		ot_used += num;
		return;
	}
	uint64_t pt = 0;
	int64_t gened = silent_ot_left();
	if(gened > 0) {
		cuda_memcpy(data.data(), ot_data.data(ot_used), gened*sizeof(blk), D2D);
		pt += gened;
	}
	int64_t round_inplace = (num-gened-M) / ot_limit;
	int64_t last_round_ot = num-gened-round_inplace*ot_limit;
	bool round_memcpy = last_round_ot>ot_limit?true:false;
	if(round_memcpy) last_round_ot -= ot_limit;

	lpn->init();

	printf("num OT requested = %ld\n", num);
	printf("OT per iteration = %ld\n", ot_limit);
	printf("iterations = %ld\n", round_inplace);

	for(int64_t i = 0; i < round_inplace; ++i) {
		Span dataSpan = data.span(pt, pt + ot_limit + M);
		extend_f2k(dataSpan);
		ot_used = ot_limit;
		pt += ot_limit;
	}
	if(round_memcpy) {
		extend_f2k();
		cuda_memcpy(data.data(pt), ot_data.data(), ot_limit*sizeof(blk), D2D);
		pt += ot_limit;
	}
	if(last_round_ot > 0) {
		extend_f2k();
		cuda_memcpy(data.data(pt), ot_data.data(), last_round_ot*sizeof(blk), D2D);
		ot_used = last_round_ot;
	}

	Log::close(mRole);
}

void FerretOTRecver::online_recver(block *data, const bool *b, int64_t length) {
	bool *bo = new bool[length];
	for(int64_t i = 0; i < length; ++i) {
		bo[i] = b[i] ^ getLSB(data[i]);
	}
	io->send_bool(bo, length*sizeof(bool));
	delete[] bo;
}

void FerretOTRecver::recv_cot(block* data, const bool * b, int64_t length) {
	Mat data_d({length});
	cuda_memcpy(data_d.data(), data, length*sizeof(blk), H2D);
	rcot(data_d);
	cuda_memcpy(data, data_d.data(), length*sizeof(blk), D2H);
	online_recver(data, b, length);
}
