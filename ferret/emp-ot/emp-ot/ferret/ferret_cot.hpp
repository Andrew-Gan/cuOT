#include "dev_layer.h"
#include <sstream>

#define TIMER_START startTime = high_resolution_clock::now();
#define TIMER_END duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count();
#define TIMER_END_COMPUTE compTime += TIMER_END;
#define TIMER_END_ONLINE onlineTime += TIMER_END;
#define TIMER_END_TOHOST h2dTime += TIMER_END;

template<typename T>
FerretCOT<T>::FerretCOT(int mlParty, int otParty, T **ios, bool malicious,
	bool run_setup, PrimalLPNParameter param, std::string pre_file, std::string log_file) {

	this->party = otParty;
	io = ios[0];
	this->ios = ios;
	this->is_malicious = malicious;
	one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
	ch[0] = zero_block;
	base_cot = new BaseCot<T>(otParty, io, malicious);
	this->param = param;
	this->extend_initialized = false;

	this->t = param.t;
	this->n = t * (1 << param.log_bin_sz);
	ot_output.resize({n});
	ch_d.resize({2});
	ch_d.read_from_cpu(ch, sizeof(ch));

	if(run_setup) {
		if(otParty == ALICE) {
			PRG prg;
			prg.random_block(&Delta);
			Delta |= 0x1;
			setup(Delta, pre_file);
		}
		else
			setup(pre_file);
	}
}

template<typename T>
FerretCOT<T>::~FerretCOT() {
	if (ot_pre_data.size() > 0) {
		block *tmp = new block[ot_pre_data.size()];
		ot_pre_data.write_to_cpu(tmp);
		if(party == ALICE)
			write_pre_data128_to_file((void*)tmp, (__uint128_t)Delta, pre_ot_filename);
		else
			write_pre_data128_to_file((void*)tmp, (__uint128_t)0, pre_ot_filename);
		delete[] tmp;
	}
	if (bo_other != nullptr)
		cuda_ipc_close_mem_handle(bo_other);
	if(pre_ot != nullptr) delete pre_ot;
	delete base_cot;
	delete pool;
	if(lpn_f2 != nullptr) delete lpn_f2;
	if(mpcot != nullptr) delete mpcot;
	if (logger != nullptr) delete logger;
}

template<typename T>
void FerretCOT<T>::extend_initialization() {
	lpn_f2 = new LpnF2<T, 10>(party, param.n, param.k, io, pool);
	mpcot = new MpcotReg<T>(party, param.n, param.t, param.log_bin_sz, pool, ios);
	if(is_malicious) mpcot->set_malicious();

	pre_ot = new OTPre<T>(io, mpcot->tree_height-1, param.t);
	M = param.k + pre_ot->n + mpcot->consist_check_cot_num;
	ot_limit = param.n - M;
	ot_used = ot_limit;
	extend_initialized = true;
}

// extend f2k in detail
template<typename T>
void FerretCOT<T>::extend(Mat &output, MpcotReg<T> &mpcot, OTPre<T> &preot, 
		LpnF2<T, 10> &lpn, Mat &ot_input) {

	if(party == ALICE) mpcot.sender_init(Delta);
	else mpcot.recver_init();
	mpcot.mpcot(output, preot, ot_input);
	lpn.compute(output, ot_input.data({(uint64_t)mpcot.consist_check_cot_num}));
}

template<typename T>
void FerretCOT<T>::extend_f2k(Mat &ot_buffer) {
	block *ot_pre_data_h = new block[ot_pre_data.size()];
	ot_pre_data.write_to_cpu(ot_pre_data_h);
	Log::start(Role(party-1), BaseOT);
	if(party == ALICE)
	    pre_ot->send_pre(ot_pre_data_h, Delta);
	else
		pre_ot->recv_pre(ot_pre_data_h);
	Log::end(Role(party-1), BaseOT);
	delete[] ot_pre_data_h;
	extend(ot_buffer, *mpcot, *pre_ot, *lpn_f2, ot_pre_data);
	ot_pre_data.read_from_gpu(ot_buffer.data({ot_limit}), M*sizeof(blk));
	ot_used = 0;
}

template<typename T>
void FerretCOT<T>::extend_f2k() {
	extend_f2k(ot_data);
}

template<typename T>
void FerretCOT<T>::setup(block Deltain, std::string pre_file) {
	this->Delta = Deltain;
	setup(pre_file);
	ch[1] = Delta;
}

template<typename T>
void FerretCOT<T>::setup(std::string pre_file) {
	if(pre_file != "") pre_ot_filename = pre_file;
	else {
		pre_ot_filename=(party==ALICE?PRE_OT_DATA_REG_SEND_FILE:PRE_OT_DATA_REG_RECV_FILE);
	}

	ThreadPool pool2(1);
	auto fut = pool2.enqueue([this](){
		extend_initialization();
	});
	ot_pre_data.resize({param.n_pre});
	bool hasfile = file_exists(pre_ot_filename), hasfile2;
	if(party == ALICE) {
		io->send_data(&hasfile, sizeof(bool));
		io->flush();
		io->recv_data(&hasfile2, sizeof(bool));
	} else {
		io->recv_data(&hasfile2, sizeof(bool));
		io->send_data(&hasfile, sizeof(bool));
		io->flush();
	}
	if(hasfile & hasfile2) {
		block *tmp = new block[param.n_pre];
		block delta = (block)read_pre_data128_from_file((void*)tmp, pre_ot_filename);
		ot_pre_data.read_from_cpu(tmp);
		delete[] tmp;
	}
	else {
		if(party == BOB) base_cot->cot_gen_pre();
		else base_cot->cot_gen_pre(Delta);

		MpcotReg<T> mpcot_init(party, param.n_pre, param.t_pre, param.log_bin_sz_pre, pool, ios);
		if(is_malicious) mpcot_init.set_malicious();
		OTPre<T> pre_ot_init(io, mpcot_init.tree_height-1, param.t_pre);
		LpnF2<T, 10> lpn(party, param.n_pre, param.k_pre, ios[0], pool);

		block *pre_data_init_h = new block[param.k_pre+mpcot_init.consist_check_cot_num];

		base_cot->cot_gen(&pre_ot_init, pre_ot_init.n);
		base_cot->cot_gen(pre_data_init_h, param.k_pre+mpcot_init.consist_check_cot_num);

		Mat pre_data_init({param.k_pre+mpcot_init.consist_check_cot_num});
		pre_data_init.read_from_cpu(pre_data_init_h, pre_data_init.size_bytes());
		delete[] pre_data_init_h;
		extend(ot_pre_data, mpcot_init, pre_ot_init, lpn, pre_data_init);
	}

	fut.get();
}

template<typename T>
void FerretCOT<T>::rcot(Mat &data, int64_t num) {
	numOT += num;
	if(ot_data.size() == 0) {
		ot_data.resize({(uint64_t)param.n});
		ot_data.clear();
	}
	if(extend_initialized == false) 
		error("Run setup before extending");
	if(num <= silent_ot_left()) {
		uint64_t start = ot_used * sizeof(block);
		ot_data.write_to_gpu(data.data(), data.size_bytes(), start);
		ot_used += num;
		return;
	}
	blk *pt = data.data();
	int64_t gened = silent_ot_left();
	if(gened > 0) {
		uint64_t cpySize = gened * sizeof(block);
		uint64_t start = ot_used * sizeof(block);
		ot_data.write_to_gpu(pt, cpySize, start);
		pt += gened;
	}
	int64_t round_inplace = (num-gened-M) / ot_limit;
	int64_t last_round_ot = num-gened-round_inplace*ot_limit;
	bool round_memcpy = last_round_ot>ot_limit?true:false;
	if(round_memcpy) last_round_ot -= ot_limit;
	for(int64_t i = 0; i < round_inplace; ++i) {
		extend_f2k(ot_output);
		uint64_t memsize = ot_limit * sizeof(block);
		ot_output.write_to_gpu(pt, memsize);
		pt += ot_limit;
		ot_used = ot_limit;
	}
	if(round_memcpy) {
		extend_f2k();
		uint64_t memsize = ot_limit * sizeof(block);
		ot_data.write_to_gpu(pt, memsize);
		pt += ot_limit;
	}
	if(last_round_ot > 0) {
		extend_f2k();
		uint64_t memsize = last_round_ot * sizeof(block);
		ot_data.write_to_gpu(pt, memsize);
        ot_used = last_round_ot;
	}
}

template<typename T>
int64_t FerretCOT<T>::silent_ot_left() {
	return ot_limit-ot_used;
}

template<typename T>
void FerretCOT<T>::write_pre_data128_to_file(void* loc, __uint128_t delta, std::string filename) {
	std::ofstream outfile(filename);
	if(outfile.is_open()) outfile.close();
	else error("create a directory to store pre-OT data");
	FileIO fio(filename.c_str(), false);
	fio.send_data(&party, sizeof(int64_t));
	if(party == ALICE) fio.send_data(&delta, 16);
	fio.send_data(&param.n, sizeof(int64_t));
	fio.send_data(&param.t, sizeof(int64_t));
	fio.send_data(&param.k, sizeof(int64_t));
	fio.send_data(loc, param.n_pre*16);
}

template<typename T>
__uint128_t FerretCOT<T>::read_pre_data128_from_file(void* pre_loc, std::string filename) {
	FileIO fio(filename.c_str(), true);
	int in_party;
	fio.recv_data(&in_party, sizeof(int64_t));
	if(in_party != party) error("wrong party");
	__uint128_t delta = 0;
	if(party == ALICE) fio.recv_data(&delta, 16);
	int64_t nin, tin, kin;
	fio.recv_data(&nin, sizeof(int64_t));
	fio.recv_data(&tin, sizeof(int64_t));
	fio.recv_data(&kin, sizeof(int64_t));
	if(nin != param.n || tin != param.t || kin != param.k)
		error("wrong parameters");
	fio.recv_data(pre_loc, param.n_pre*16);
	std::remove(filename.c_str());
	return delta;
}

template<typename T>
int64_t FerretCOT<T>::byte_memory_need_inplace(int64_t ot_need) {
	int64_t round = (ot_need - 1) / ot_limit;
	return round * ot_limit + param.n;
}

template<typename T>
void FerretCOT<T>::online_sender(block *data, int64_t length) {
	bool newMemHandle = bo.resize(length);
	ios[0]->send_data(&newMemHandle, sizeof(newMemHandle));
	if (newMemHandle) {
		uint8_t handle[64];
		bo.get_mem_handle(handle);
		ios[0]->send_data(handle, sizeof(handle));
		ios[0]->flush();
	}
	bool dataWritten = false;
	ios[0]->recv_data(&dataWritten, sizeof(dataWritten));
	cuda_online_sender(bo, ch_d, length_data, length);
    if (data == nullptr) return;
	length_data.write_to_cpu(data);
}

template<typename T>
void FerretCOT<T>::online_recver(block *data, const bool *b, int64_t length) {
	bool newMemHandle = false;
	ios[0]->recv_data(&newMemHandle, sizeof(newMemHandle));
	if (newMemHandle) {
		if (bo_other != nullptr)
			cuda_ipc_close_mem_handle(bo_other);
		uint8_t handle[64];
		ios[0]->recv_data(handle, sizeof(handle));
		cuda_ipc_open_mem_handle(&bo_other, handle);
	}
	bo.resize(length);
	b_d.resize(length);
	b_d.read_from_cpu(b, length);
	cuda_online_recver(bo, bo_other, b_d, length_data, length);
	bool dataWritten = true;
	ios[0]->send_data(&dataWritten, sizeof(dataWritten));
	ios[0]->flush();
    if (data == nullptr) return;
	length_data.write_to_cpu(data);
}

template<typename T>
void FerretCOT<T>::send_cot(block * data, int64_t length) {
	length_data.resize({(uint64_t)length});
	rcot(length_data, length);
	online_sender(data, length);
}

template<typename T>
void FerretCOT<T>::recv_cot(block* data, const bool * b, int64_t length) {
	length_data.resize({(uint64_t)length});
	rcot(length_data, length);
	online_recver(data, b, length);
}
