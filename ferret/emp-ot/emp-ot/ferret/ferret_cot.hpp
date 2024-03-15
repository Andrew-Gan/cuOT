template<typename T>
FerretCOT<T>::FerretCOT(int party, T *ios,
		bool malicious, bool run_setup, PrimalLPNParameter param, std::string pre_file) {
	
	this->party = party;
	io = ios;
	this->ios = ios;
	this->is_malicious = malicious;
	one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
	ch[0] = zero_block;
	base_cot = new BaseCot<T>(party, io, malicious);
	// pool = new ThreadPool(threads);
	this->param = param;
	this->extend_initialized = false;
	cuda_init(party);
	ot_pre_data.resize({param.n_pre});

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
	cuda_memcpy(tmp, ot_pre_data.data(), param.n_pre*sizeof(block), D2H);
	if (ot_pre_data.size() > 0) {
		if(party == ALICE) write_pre_data128_to_file((void*)tmp, (__uint128_t)Delta, pre_ot_filename);
		else write_pre_data128_to_file((void*)tmp, (__uint128_t)0, pre_ot_filename);
	}
	delete[] tmp;
	delete base_cot;
	if(lpn_f2 != nullptr) delete lpn_f2;
	if(mpcot != nullptr) delete mpcot;
}

template<typename T>
void FerretCOT<T>::extend_initialization() {
	cuda_init(party);
	lpn_f2 = new LpnF2<T, 10>(party, param.n, param.k, io);
	mpcot = new MpcotReg<T>(party, param.n, param.t, param.log_bin_sz, ios);
	if(is_malicious) mpcot->set_malicious();

	pre_ot = new OTPre<T>(io, mpcot->tree_n*(mpcot->tree_height-1), 1);
	M = param.k + pre_ot->n + mpcot->consist_check_cot_num;
	ot_limit = param.n - M;
	ot_used = ot_limit;
	extend_initialized = true;
}

// extend f2k in detail
template<typename T>
void FerretCOT<T>::extend(Span *ot_output, MpcotReg<T> *mpcot, OTPre<T> *preot,
		LpnF2<T, 10> *lpn, Mat &ot_input) {

	blk Delta_blk;
	memcpy(&Delta_blk, &Delta, sizeof(blk));
	if(party == ALICE) mpcot->sender_init(Delta_blk);
	else mpcot->recver_init();

	mpcot->mpcot(ot_output, preot, ot_input);

	Log::start(party-1, LPN);
	Span kSpan = ot_input.span(mpcot->consist_check_cot_num);
	Log::mem(party-1, LPN);
	lpn->compute(ot_output, kSpan);
	Log::mem(party-1, LPN);
	Log::end(party-1, LPN);
}

template<typename T>
void FerretCOT<T>::extend(Mat *ot_output, MpcotReg<T> *mpcot, OTPre<T> *preot,
		LpnF2<T, 10> *lpn, Mat &ot_input) {

	Span otSpan[NGPU];
	for (int gpu = 0; gpu < NGPU; gpu++) {
		otSpan[gpu] = Span(ot_output, 0, ot_output[gpu].size());
	}
	extend(otSpan, mpcot, preot, lpn, ot_input);
}

// extend f2k (customized location)
template<typename T>
void FerretCOT<T>::extend_f2k(Span *ot_buffer) {
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
	cuda_memcpy(ot_pre_data.data(), ot_buffer.data(ot_limit), M*sizeof(block), D2D);

	ot_used = 0;

}

// extend f2k
template<typename T>
void FerretCOT<T>::extend_f2k() {
	Span tmp = ot_data.span();
	extend_f2k(tmp);
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

	cuda_init(party);

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

		MpcotReg<T> mpcot_ini(party, param.n_pre, param.t_pre, param.log_bin_sz_pre, ios);
		if(is_malicious) mpcot_ini.set_malicious();
		OTPre<T> pre_ot_ini(ios, mpcot_ini.tree_n*(mpcot_ini.tree_height-1), 1);
		LpnF2<T, 10> lpn(party, param.n_pre, param.k_pre, io);

		block *pre_data_ini = new block[param.k_pre+mpcot_ini.consist_check_cot_num];
		ot_pre_data.clear();

		base_cot->cot_gen(&pre_ot_ini, pre_ot_ini.n);
		base_cot->cot_gen(pre_data_ini, param.k_pre+mpcot_ini.consist_check_cot_num);
		Mat tmp({param.k_pre+mpcot_ini.consist_check_cot_num});
		cuda_memcpy(tmp.data(), pre_data_ini, tmp.size_bytes(), H2D);
		lpn.init();
		extend(ot_pre_data, &mpcot_ini, &pre_ot_ini, &lpn, tmp);
		delete[] pre_data_ini;
	}

	fut.get();
}

template<typename T>
void FerretCOT<T>::rcot(Mat &data) {
	int64_t num = (int64_t) data.size();
	if(ot_data.size() == 0) {
		ot_data.resize({param.n});
		ot_data.clear();
	}
	if(extend_initialized == false)
		error("Run setup before extending");
	if(num <= silent_ot_left()) {
		cuda_memcpy(data.data(), ot_data.data(ot_used), num*sizeof(block), D2D);
		ot_used += num;
		return;
	}
	uint64_t pt = 0;
	int64_t gened = silent_ot_left();
	if(gened > 0) {
		cuda_memcpy(data.data(), ot_data.data(ot_used), gened*sizeof(block), D2D);
		pt += gened;
	}
	int64_t round_inplace = (num-gened-M) / ot_limit;
	int64_t last_round_ot = num-gened-round_inplace*ot_limit;
	bool round_memcpy = last_round_ot>ot_limit?true:false;
	if(round_memcpy) last_round_ot -= ot_limit;

	lpn_f2->init();

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

	Log::close(party-1);
}

template<typename T>
int64_t FerretCOT<T>::silent_ot_left() {
	return ot_limit-ot_used;
}

template<typename T>
void FerretCOT<T>::write_pre_data128_to_file(void* loc, __uint128_t delta, std::string filename) {
	std::ofstream outfile(filename);
	if(outfile.is_open()) outfile.close();
	else {
		printf("Attempted to create directory at %s\n", filename.c_str());
		error("create a directory to store pre-OT data");
	}
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
int64_t FerretCOT<T>::rcot_inplace(Mat *ot_buffer) {
	int64_t byte_space = (int64_t)ot_buffer[0].size();
	if(byte_space < param.n) error("space not enough");
	if((byte_space - M) % ot_limit != 0) error("call byte_memory_need_inplace \
			to get the correct length of memory space");
	int64_t ot_output_n = byte_space - M;
	int64_t round = ot_output_n / ot_limit;
	// block *pt = ot_buffer;

	block *tmp = new block[M];

	for (int gpu = 0; gpu < NGPU; gpu++) {
		for(int64_t i = 0; i < round; ++i) {
			if(party == ALICE) {
				cuda_memcpy(tmp, ot_pre_data.data(), M*sizeof(block), D2H);
				pre_ot->send_pre(tmp, Delta);
			}
			else {
				pre_ot->recv_pre(tmp);
				cuda_memcpy(ot_pre_data[gpu].data(), tmp, M*sizeof(block), H2D);
			}
			extend(ot_buffer, mpcot, pre_ot, lpn_f2, ot_pre_data);
			// pt += ot_limit;
			// memcpy(ot_pre_data, ot_buffer + ot_limit, M*sizeof(block));
			cuda_memcpy(ot_pre_data[gpu].data(), ot_buffer[gpu].data(ot_limit*(i+1)), M*sizeof(block), D2D);
		}
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
	Mat data_d({length});
	cuda_memcpy(data_d.data(), data, length*sizeof(blk), H2D);
	rcot(data_d);
	cuda_memcpy(data, data_d.data(), length*sizeof(blk), D2H);
	online_sender(data, length);
}

template<typename T>
void FerretCOT<T>::recv_cot(block* data, const bool * b, int64_t length) {
	Mat data_d({length});
	cuda_memcpy(data_d.data(), data, length*sizeof(blk), H2D);
	rcot(data_d);
	cuda_memcpy(data, data_d.data(), length*sizeof(blk), D2H);
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
	cuda_memcpy(array + sizeof(int64_t) * 4 + sizeof(block), ot_pre_data.data(), sizeof(block)*param.n_pre, D2H);
	ot_pre_data.clear();
}

template<typename T>
int FerretCOT<T>::disassemble_state(const void * data, int64_t size) {
	const unsigned char * array = (const unsigned char * )data;
	int64_t n2 = 0, t2 = 0, k2 = 0, party2 = 0;
	// ot_pre_data = new block[param.n_pre];
	ot_pre_data.resize({param.n_pre});
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

