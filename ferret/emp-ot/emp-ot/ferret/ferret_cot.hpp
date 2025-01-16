#include "dev_layer.h"
#include <sstream>

#define TIMER_START startTime = high_resolution_clock::now();
#define TIMER_END duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count();
#define TIMER_END_COMPUTE compTime += TIMER_END;
#define TIMER_END_ONLINE onlineTime += TIMER_END;
#define TIMER_END_TOHOST h2dTime += TIMER_END;

template<typename T>
FerretCOT<T>::FerretCOT(int mlParty, int otParty, int ngpu, T **ios,
	bool malicious, bool run_setup, PrimalLPNParameter param, std::string pre_file, std::string log_file) {
	
	assert(ngpu > 0);
	this->ngpu = ngpu;
	this->party = otParty;
	io = ios[0];
	this->ios = ios;
	this->is_malicious = malicious;
	one = makeBlock(0xFFFFFFFFFFFFFFFFLL,0xFFFFFFFFFFFFFFFELL);
	ch[0] = zero_block;
	base_cot = new BaseCot<T>(otParty, io, malicious);
	pool = new ThreadPool(ngpu);
	this->param = param;
	this->extend_initialized = false;

	ch_d = new Mat[ngpu];
	ot_output = new Mat[ngpu];
	ot_data = new Mat[ngpu];
	ot_pre_data = new Mat[ngpu];
    bo = new GPUdata[ngpu];
	b_d = new GPUdata[ngpu];
	length_data = new Mat[ngpu];
	bo_other = new void*[ngpu];
	memset(bo_other, 0, sizeof(void*) * ngpu);

	tPerGPU = (param.t + (ngpu-1)) / ngpu;
	nPerGPU = tPerGPU * (1 << param.log_bin_sz);
	GPU_PARALLEL_FOR(
		ot_output[i].resize({nPerGPU});
		ch_d[i].resize({2});
		ch_d[i].read_from_cpu(ch, sizeof(ch));
    )

	if(run_setup) {
		if(otParty == ALICE) {
			PRG prg;
			prg.random_block(&Delta);
			Delta = Delta & one;
			Delta = Delta ^ 0x1;
			setup(Delta, pre_file);
		}
		else
			setup(pre_file);
	}
	// Log::start(Role(otParty-1), Neural);
}

template<typename T>
FerretCOT<T>::~FerretCOT() {
	// Log::end(Role(party-1), Neural);
    GPU_PARALLEL_FOR(
        if (ot_pre_data[i].size() > 0) {
            block *tmp = new block[ot_pre_data[i].size()];
            ot_pre_data[i].write_to_cpu(tmp);
            std::string file = pre_ot_filename + std::to_string(i);
            if(party == ALICE)
                write_pre_data128_to_file((void*)tmp, (__uint128_t)Delta, file);
            else
                write_pre_data128_to_file((void*)tmp, (__uint128_t)0, file);
            delete[] tmp;
        }
		if (bo_other[i] != nullptr)
			cuda_ipc_close_mem_handle(bo_other[i]);
    )
	if(pre_ot != nullptr) delete pre_ot;
	delete base_cot;
	delete pool;
	if(lpn_f2 != nullptr) delete lpn_f2;
	if(mpcot != nullptr) delete mpcot;
	delete[] ch_d;
	delete[] ot_output;
	delete[] ot_data;
	delete[] ot_pre_data;
    delete[] bo;
	delete[] b_d;
	delete[] length_data;
	delete[] bo_other;

	if (logger != nullptr) {
		delete logger;
	}
}

template<typename T>
void FerretCOT<T>::extend_initialization() {
	lpn_f2 = new LpnF2<T, 10>(party, param.n, param.k, io, pool, ngpu);
	mpcot = new MpcotReg<T>(party, ngpu, param.n, param.t, param.log_bin_sz, pool, ios);
	if(is_malicious) mpcot->set_malicious();

	pre_ot = new OTPre<T>(io, mpcot->tree_height-1, param.t);
	M = param.k + pre_ot->n + mpcot->consist_check_cot_num;
	ot_limit = param.n - M;
	ot_used = ot_limit;
	extend_initialized = true;
}

// extend f2k in detail
template<typename T>
void FerretCOT<T>::extend(MpcotReg<T> *mpcot, OTPre<T> *preot, 
		LpnF2<T, 10> *lpn, Mat *ot_input) {

	if(party == ALICE) mpcot->sender_init(Delta);
	else mpcot->recver_init();
	Log::start(Role(party-1), SeedExp);
	mpcot->mpcot(ot_output, preot, ot_input);
	Log::end(Role(party-1), SeedExp);
	Log::start(Role(party-1), LPN);
    blk **kk = new blk*[ngpu];
    for (int i = 0; i < ngpu; i++) {
        kk[i] = ot_input[i].data({(uint64_t)mpcot->consist_check_cot_num});
	}
	lpn->compute(ot_output, kk);
    delete[] kk;
	Log::end(Role(party-1), LPN);
}

template<typename T>
void FerretCOT<T>::extend_f2k() {
	block *tmp = new block[ot_pre_data[0].size()];
	ot_pre_data[0].write_to_cpu(tmp);
	Log::start(Role(party-1), BaseOT);
	if(party == ALICE)
	    pre_ot->send_pre(tmp, Delta);
	else
		pre_ot->recv_pre(tmp);
	Log::end(Role(party-1), BaseOT);
	delete[] tmp;
	extend(mpcot, pre_ot, lpn_f2, ot_pre_data);
    GPU_PARALLEL_FOR(
        ot_pre_data[i].read_from_gpu(
            ot_output[i].data({ot_output[i].size() - M}), M*sizeof(block)
        );
    )
	ot_used = 0;
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
    GPU_PARALLEL_FOR(
	    ot_pre_data[i].resize({param.n_pre});
    )
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
        GPU_PARALLEL_FOR(
            block *tmp = new block[param.n_pre];
            std::string file = pre_ot_filename + std::to_string(i);
            block delta = (block)read_pre_data128_from_file((void*)tmp, file);
            if (i == 0) Delta = delta;
            ot_pre_data[i].read_from_cpu(tmp);
		    delete[] tmp;
        )
	}
	else {
		if(party == BOB) base_cot->cot_gen_pre();
		else base_cot->cot_gen_pre(Delta);
		io->flush();

		MpcotReg<T> mpcot_ini(party, ngpu, param.n_pre, param.t_pre, param.log_bin_sz_pre, pool, ios);
		if(is_malicious) mpcot_ini.set_malicious();
		OTPre<T> pre_ot_ini(io, mpcot_ini.tree_height-1, param.t_pre);
		LpnF2<T, 10> lpn(party, param.n_pre, param.k_pre, ios[0], pool, ngpu);

		block *pre_data_ini = new block[param.k_pre+mpcot_ini.consist_check_cot_num];

		base_cot->cot_gen(&pre_ot_ini, pre_ot_ini.n);
		base_cot->cot_gen(pre_data_ini, param.k_pre+mpcot_ini.consist_check_cot_num);
		io->flush();

		Mat *tmp = new Mat[ngpu];
		GPU_PARALLEL_FOR(
			tmp[i].resize({(param.k_pre+mpcot_ini.consist_check_cot_num)});
			tmp[i].read_from_cpu(pre_data_ini, tmp[i].size_bytes());
		)
		delete[] pre_data_ini;
		extend(&mpcot_ini, &pre_ot_ini, &lpn, tmp);
		delete[] tmp;
		GPU_PARALLEL_FOR(
			int nPrePerGPU = param.n_pre/ngpu;
			uint64_t cpySize = nPrePerGPU * sizeof(blk);
			for (int j = 0; j < ngpu; j++) {
				ot_output[i].write_to_gpu(ot_pre_data[j].data({i*nPrePerGPU}), cpySize, 0, j);
			}
		)
	}

	fut.get();
}

template<typename T>
void FerretCOT<T>::rcot(Mat *data, int64_t num) {
	TIMER_START
	numOT += num;
    GPU_PARALLEL_FOR(
        if(ot_data[i].size() == 0) {
            ot_data[i].resize({(uint64_t)param.n / ngpu});
            ot_data[i].clear();
		}
    )
	if(extend_initialized == false) 
		error("Run setup before extending");
	if(num <= silent_ot_left()) {
        GPU_PARALLEL_FOR(
            uint64_t start = ot_used * sizeof(block) / ngpu;
            ot_data[i].write_to_gpu(data[i].data(), data[i].size_bytes(), start);
        )
		ot_used += num;
		TIMER_END_COMPUTE
		return;
	}
	blk **pt = new blk*[ngpu];
	for (int i = 0; i < ngpu; i++) pt[i] = data[i].data();
	int64_t gened = silent_ot_left();
	if(gened > 0) {
        GPU_PARALLEL_FOR(
            uint64_t cpySize = (gened / ngpu) * sizeof(block);
            uint64_t start = (ot_used / ngpu) * sizeof(block);
            ot_data[i].write_to_gpu(pt[i], cpySize, start);
			pt[i] += gened / ngpu;
        )
	}
	int64_t round_inplace = (num-gened-M) / ot_limit;
	int64_t last_round_ot = num-gened-round_inplace*ot_limit;
	bool round_memcpy = last_round_ot>ot_limit?true:false;
	if(round_memcpy) last_round_ot -= ot_limit;
	for(int64_t i = 0; i < round_inplace; ++i) {
		extend_f2k();
        GPU_PARALLEL_FOR(
            uint64_t memsize = (ot_limit / ngpu) * sizeof(block);
            ot_output[i].write_to_gpu(pt[i], memsize);
			pt[i] += ot_limit / ngpu;
        )
		ot_used = ot_limit;
	}
	if(round_memcpy) {
		extend_f2k();
        GPU_PARALLEL_FOR(
		    ot_data[i] = ot_output[i];
            uint64_t memsize = (ot_limit / ngpu) * sizeof(block);
            ot_data[i].write_to_gpu(pt[i], memsize);
			pt[i] += ot_limit / ngpu;
        )
	}
	if(last_round_ot > 0) {
		extend_f2k();
        GPU_PARALLEL_FOR(
            ot_data[i] = ot_output[i];
            uint64_t memsize = (last_round_ot / ngpu) * sizeof(block);
            ot_data[i].write_to_gpu(pt[i], memsize);
        )
        ot_used = last_round_ot;
	}
	delete[] pt;
	TIMER_END_COMPUTE
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
	TIMER_START
    int64_t lengthPerGPU = length / ngpu;
    GPU_PARALLEL_FOR(
        bool newMemHandle = bo[i].resize(lengthPerGPU);
        ios[i]->send_data(&newMemHandle, sizeof(newMemHandle));
		ios[i]->flush();
        if (newMemHandle) {
            uint8_t handle[64];
            bo[i].get_mem_handle(handle);
            ios[i]->send_data(handle, sizeof(handle));
			ios[i]->flush();
        }
        bool dataWritten = false;
        ios[i]->recv_data(&dataWritten, sizeof(dataWritten));
        cuda_online_sender(bo[i], ch_d[i], length_data[i], lengthPerGPU);
    )
    TIMER_END_ONLINE
    if (data == nullptr) return;
	TIMER_START
    GPU_PARALLEL_FOR(
        block *mydata = data + i * lengthPerGPU;
        length_data[i].write_to_cpu(mydata);
    )
	TIMER_END_TOHOST
}

template<typename T>
void FerretCOT<T>::online_recver(block *data, const bool *b, int64_t length) {
	TIMER_START
    int64_t lengthPerGPU = length / ngpu;
    GPU_PARALLEL_FOR(
        bool newMemHandle = false;
        ios[i]->recv_data(&newMemHandle, sizeof(newMemHandle));
        if (newMemHandle) {
            if (bo_other[i] != nullptr)
                cuda_ipc_close_mem_handle(bo_other[i]);
            uint8_t handle[64];
            ios[i]->recv_data(handle, sizeof(handle));
            cuda_ipc_open_mem_handle(&bo_other[i], handle);
        }
        bo[i].resize(lengthPerGPU);
        b_d[i].resize(lengthPerGPU);
        b_d[i].read_from_cpu(b, lengthPerGPU);
        cuda_online_recver(bo[i], bo_other[i], b_d[i], length_data[i], lengthPerGPU);
        bool dataWritten = true;
        ios[i]->send_data(&dataWritten, sizeof(dataWritten));
		ios[i]->flush();
    )
	TIMER_END_ONLINE
    if (data == nullptr) return;
	TIMER_START
	GPU_PARALLEL_FOR(
        block *mydata = data + i * lengthPerGPU;
        length_data[i].write_to_cpu(mydata);
    )
	TIMER_END_TOHOST
}

template<typename T>
void FerretCOT<T>::send_cot(block * data, int64_t length) {
	// Log::end(Role(party-1), Neural);
	TIMER_START
    GPU_PARALLEL_FOR(
	    length_data[i].resize({(uint64_t)length / ngpu});
    )
	TIMER_END_COMPUTE
	rcot(length_data, length);
	online_sender(data, length);
	// Log::start(Role(party-1), Neural);
}

template<typename T>
void FerretCOT<T>::recv_cot(block* data, const bool * b, int64_t length) {
	// Log::end(Role(party-1), Neural);
	TIMER_START
	GPU_PARALLEL_FOR(
	    length_data[i].resize({(uint64_t)length / ngpu});
    )
	TIMER_END_COMPUTE
	rcot(length_data, length);
	online_recver(data, b, length);
	// Log::start(Role(party-1), Neural);
}
