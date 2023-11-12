#include "cuda_layer.h"

template<typename T>
FerretCOT<T>::FerretCOT(int party, int threads, T **ios,
		bool malicious, bool run_setup, PrimalLPNParameter param, std::string pre_file) {
	this->party = party;
	this->threads = threads;
	io = ios[0];
	this->ios = ios;
	this->is_malicious = malicious;
	one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
	ch[0] = zero_block;
	base_cot = new BaseCot<T>(party, io, malicious);
	pool = new ThreadPool(threads);
	this->param = param;

	this->extend_initialized = false;

	if(run_setup) {
		if(party == ALICE) {
			PRG prg;
			prg.random_block(&Delta);
			Delta = Delta & one;
			Delta = Delta ^ 0x1;
			setup(Delta, pre_file);
		} else setup(pre_file);
	}
}

template<typename T>
FerretCOT<T>::~FerretCOT() {
	block *tmp = new block[param.n_pre];
	cuda_memcpy(tmp, ot_pre_data.data(), param.n_pre*sizeof(blk), D2H);
	if (ot_pre_data.size() > 0) {
		if(party == ALICE) write_pre_data128_to_file((void*)tmp, (__uint128_t)Delta, pre_ot_filename);
		else write_pre_data128_to_file((void*)tmp, (__uint128_t)0, pre_ot_filename);
		delete[] tmp;
	}
	delete base_cot;
	delete pool;
	if(lpn_f2 != nullptr) delete lpn_f2;
	if(mpcot != nullptr) delete mpcot;
}

template<typename T>
void FerretCOT<T>::extend_initialization() {
	lpn_f2 = new LpnF2<T, 10>(party, param.n, param.k, pool, io, pool->size());
	mpcot = new MpcotReg<T>(party, threads, param.n, param.t, param.log_bin_sz, pool, ios);
	if(is_malicious) mpcot->set_malicious();

	pre_ot = new OTPre<T>(io, mpcot->tree_height-1, mpcot->tree_n);
	M = param.k + pre_ot->n + mpcot->consist_check_cot_num;
	ot_limit = param.n - M;
	ot_used = ot_limit;
	extend_initialized = true;
}

// extend f2k in detail
template<typename T>
void FerretCOT<T>::extend(vec &ot_output, MpcotReg<T> *mpcot, OTPre<T> *preot,
		LpnF2<T, 10> *lpn, vec &ot_input) {

	cuda_init();

	struct timespec t[4];

	clock_gettime(CLOCK_MONOTONIC, &t[0]);

	blk Delta_blk;
	memcpy(&Delta_blk, &Delta, sizeof(blk));

	if(party == ALICE) mpcot->sender_init(Delta_blk);
	else mpcot->recver_init();

	clock_gettime(CLOCK_MONOTONIC, &t[1]);

	mpcot->mpcot(ot_output, preot, ot_input);

	clock_gettime(CLOCK_MONOTONIC, &t[2]);

	lpn->compute(ot_output, ot_input.data(mpcot->consist_check_cot_num));

	clock_gettime(CLOCK_MONOTONIC, &t[3]);

	printf("\n");
	const char* actionStr[] = {"RoleInit", "COT-TreeExp", "LPN-MatMult"};
	for (int i = 0; i < 3; i++) {
		float duration = (t[i+1].tv_sec-t[i].tv_sec) * 1000.0f;
		duration += (t[i+1].tv_nsec-t[i].tv_nsec) / 1000000.0f;
		printf("extend %d %s: %.2f ms\n", party, actionStr[i], duration);
	}
}

// extend f2k (customized location)
template<typename T>
void FerretCOT<T>::extend_f2k(vec &ot_buffer) {
	block *tmp = new block[param.n_pre];

	if(party == ALICE) {
		cuda_memcpy(tmp, ot_pre_data.data(), M*sizeof(block), D2H);
		pre_ot->send_pre(tmp, Delta);
	}
	else {
		pre_ot->recv_pre(tmp);
		cuda_memcpy(ot_pre_data.data(), tmp, M*sizeof(block), H2D);
	}

	extend(ot_buffer, mpcot, pre_ot, lpn_f2, ot_pre_data);
	// memcpy(ot_pre_data, ot_buffer+ot_limit, M*sizeof(block));
	cuda_memcpy(ot_pre_data.data(), ot_buffer.data(ot_limit), M*sizeof(block), D2D);

	ot_used = 0;
}

// extend f2k
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

	// ot_pre_data = new block[param.n_pre];
	ot_pre_data.resize(param.n_pre);
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
		block *buffer = new block[param.n_pre];
		Delta = (block)read_pre_data128_from_file((void*)buffer, pre_ot_filename);
		cuda_memcpy(ot_pre_data.data(), buffer, param.n_pre*sizeof(blk), H2D);
		delete[] buffer;
	} else {
		if(party == BOB) base_cot->cot_gen_pre();
		else base_cot->cot_gen_pre(Delta);

		MpcotReg<T> mpcot_ini(party, threads, param.n_pre, param.t_pre, param.log_bin_sz_pre, pool, ios);
		if(is_malicious) mpcot_ini.set_malicious();
		OTPre<T> pre_ot_ini(ios[0], mpcot_ini.tree_height-1, mpcot_ini.tree_n);
		LpnF2<T, 10> lpn(party, param.n_pre, param.k_pre, pool, io, pool->size());

		block *pre_data_ini = new block[param.k_pre+mpcot_ini.consist_check_cot_num];
		// memset(this->ot_pre_data, 0, param.n_pre*16);
		ot_pre_data.clear();

		base_cot->cot_gen(&pre_ot_ini, pre_ot_ini.n);
		base_cot->cot_gen(pre_data_ini, param.k_pre+mpcot_ini.consist_check_cot_num);
		vec tmp(param.k_pre+mpcot_ini.consist_check_cot_num);
		cuda_memcpy(tmp.data(), pre_data_ini, tmp.size_bytes(), H2D);
		extend(ot_pre_data, &mpcot_ini, &pre_ot_ini, &lpn, tmp);
		delete[] pre_data_ini;
	}

	fut.get();
}

template<typename T>
void FerretCOT<T>::rcot(block *data, int64_t num) {
	if(ot_data.size() == 0) {
		// ot_data = new block[param.n];
		ot_data.resize(param.n);
		// memset(ot_data, 0, param.n*sizeof(block));
		ot_data.clear();
	}
	if(extend_initialized == false)
		error("Run setup before extending");
	if(num <= silent_ot_left()) {
		// memcpy(data, ot_data+ot_used, num*sizeof(block));
		cuda_memcpy(data, ot_data.data(ot_used), num*sizeof(block), D2H);
		ot_used += num;
		return;
	}
	block *pt = data;
	int64_t gened = silent_ot_left();
	if(gened > 0) {
		// memcpy(pt, ot_data+ot_used, gened*sizeof(block));
		cuda_memcpy(pt, ot_data.data(ot_used), gened*sizeof(block), D2H);
		pt += gened;
	}
	int64_t round_inplace = (num-gened-M) / ot_limit;
	int64_t last_round_ot = num-gened-round_inplace*ot_limit;
	bool round_memcpy = last_round_ot>ot_limit?true:false;
	if(round_memcpy) last_round_ot -= ot_limit;
	vec tmp(ot_limit);
	for(int64_t i = 0; i < round_inplace; ++i) {
		cuda_memcpy(tmp.data(), pt, ot_limit*sizeof(blk), H2D);
		extend_f2k(tmp);
		ot_used = ot_limit;
		pt += ot_limit;
	}
	if(round_memcpy) {
		extend_f2k();
		// memcpy(pt, ot_data, ot_limit*sizeof(block));
		cuda_memcpy(pt, ot_data.data(), ot_limit*sizeof(block), D2H);
		pt += ot_limit;
	}
	if(last_round_ot > 0) {
		extend_f2k();
		// memcpy(pt, ot_data, last_round_ot*sizeof(block));
		cuda_memcpy(pt, ot_data.data(), last_round_ot*sizeof(block), D2H);
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
	__uint128_t delta;
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

// extend f2k (benchmark)
// parameter "length" should be the return of "byte_memory_need_inplace"
// output the number of COTs that can be used
template<typename T>
int64_t FerretCOT<T>::rcot_inplace(vec &ot_buffer, int64_t byte_space) {
	if(byte_space < param.n) error("space not enough");
	if((byte_space - M) % ot_limit != 0) error("call byte_memory_need_inplace \
			to get the correct length of memory space");
	int64_t ot_output_n = byte_space - M;
	int64_t round = ot_output_n / ot_limit;
	// block *pt = ot_buffer;

	block *tmp = new block[M];

	for(int64_t i = 0; i < round; ++i) {
		if(party == ALICE) {
			cuda_memcpy(tmp, ot_pre_data.data(), M*sizeof(block), D2H);
		    pre_ot->send_pre(tmp, Delta);
		}
		else {
			pre_ot->recv_pre(tmp);
			cuda_memcpy(ot_pre_data.data(), tmp, M*sizeof(block), H2D);
		}
		extend(ot_buffer, mpcot, pre_ot, lpn_f2, ot_pre_data);
		// pt += ot_limit;
		// memcpy(ot_pre_data, ot_buffer + ot_limit, M*sizeof(block));
		cuda_memcpy(ot_pre_data.data(), ot_buffer.data(ot_limit*(i+1)), M*sizeof(block), D2D);
	}

	delete[] tmp;

	return ot_output_n;
}

template<typename T>
void FerretCOT<T>::online_sender(block *data, int64_t length) {
	bool *bo = new bool[length];
	io->recv_bool(bo, length*sizeof(bool));
	for(int64_t i = 0; i < length; ++i) {
		data[i] = data[i] ^ ch[bo[i]];
	}
	delete[] bo;
}

template<typename T>
void FerretCOT<T>::online_recver(block *data, const bool *b, int64_t length) {
	bool *bo = new bool[length];
	for(int64_t i = 0; i < length; ++i) {
		bo[i] = b[i] ^ getLSB(data[i]);
	}
	io->send_bool(bo, length*sizeof(bool));
	delete[] bo;
}

template<typename T>
void FerretCOT<T>::send_cot(block * data, int64_t length) {
	rcot(data, length);
	online_sender(data, length);
}

template<typename T>
void FerretCOT<T>::recv_cot(block* data, const bool * b, int64_t length) {
	rcot(data, length);
	online_recver(data, b, length);
}

template<typename T>
void FerretCOT<T>::assemble_state(void * data, int64_t size) {
	unsigned char * array = (unsigned char * )data;
	int64_t party_tmp = party;
	memcpy(array, &party_tmp, sizeof(int64_t));
	memcpy(array + sizeof(int64_t), &param.n, sizeof(int64_t));
	memcpy(array + sizeof(int64_t) * 2, &param.t, sizeof(int64_t));
	memcpy(array + sizeof(int64_t) * 3, &param.k, sizeof(int64_t));
	memcpy(array + sizeof(int64_t) * 4, &Delta, sizeof(block));
	// memcpy(array + sizeof(int64_t) * 4 + sizeof(block), ot_pre_data, sizeof(block)*param.n_pre);
	cuda_memcpy(array + sizeof(int64_t) * 4 + sizeof(block), ot_pre_data.data(), sizeof(block)*param.n_pre, D2H);
	// if (ot_pre_data!= nullptr)
	// 	delete[] ot_pre_data;
	// ot_pre_data = nullptr;
	ot_pre_data.clear();
}

template<typename T>
int FerretCOT<T>::disassemble_state(const void * data, int64_t size) {
	const unsigned char * array = (const unsigned char * )data;
	int64_t n2 = 0, t2 = 0, k2 = 0, party2 = 0;
	// ot_pre_data = new block[param.n_pre];
	ot_pre_data.resize(param.n_pre);
	memcpy(&party2, array, sizeof(int64_t));
	memcpy(&n2, array + sizeof(int64_t), sizeof(int64_t));
	memcpy(&t2, array + sizeof(int64_t) * 2, sizeof(int64_t));
	memcpy(&k2, array + sizeof(int64_t) * 3, sizeof(int64_t));
	if(party2 != party or n2 != param.n or t2 != param.t or k2 != param.k) {
		return -1;
	}
	memcpy(&Delta, array + sizeof(int64_t) * 4, sizeof(block));
	// memcpy(ot_pre_data, array + sizeof(int64_t) * 4 + sizeof(block), sizeof(block)*param.n_pre);
	cuda_memcpy(ot_pre_data.data(), array + sizeof(int64_t) * 4 + sizeof(block), sizeof(block)*param.n_pre, H2D);

	extend_initialization();
	ch[1] = Delta;
	return 0;
}

template<typename T>
int64_t FerretCOT<T>::state_size() {
	return sizeof(int64_t) * 4 + sizeof(block) + sizeof(block)*param.n_pre;
}

