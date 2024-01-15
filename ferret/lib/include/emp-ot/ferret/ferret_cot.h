#ifndef EMP_FERRET_COT_H_
#define EMP_FERRET_COT_H_
#include "emp-ot/ferret/mpcot_reg.h"
#include "emp-ot/ferret/base_cot.h"
#include "emp-ot/ferret/lpn_f2.h"
#include "emp-ot/ferret/constants.h"

#include "cuda_layer.h"
#include "logger.h"

namespace emp {

/*
 * Ferret COT binary version
 * [REF] Implementation of "Ferret: Fast Extension for coRRElated oT with small communication"
 * https://eprint.iacr.org/2020/924.pdf
 *
 */
template<typename T>
class FerretCOT: public COT<T> {
public:
	using COT<T>::io;
	using COT<T>::Delta;

	PrimalLPNParameter param;
	int64_t ot_used, ot_limit;

	FerretCOT(int party, T *ios, bool malicious = false, bool run_setup = true,
PrimalLPNParameter param = ferret_cuda, std::string pre_file="");


	~FerretCOT();

	void setup(block Deltain, std::string pre_file = "");

	void setup(std::string pre_file = "");

	void send_cot(block * data, int64_t length) override;

	void recv_cot(block* data, const bool * b, int64_t length) override;

	void rcot(Vec &data);

	int64_t rcot_inplace(Vec &ot_buffer);

	int64_t byte_memory_need_inplace(int64_t ot_need);

	void assemble_state(void * data, int64_t size);

	int disassemble_state(const void * data, int64_t size);

	int64_t state_size();
private:
	block ch[2];

	T *ios;
	int party;
	int64_t M;
	bool is_malicious;
	bool extend_initialized;

	block one;

	Vec ot_pre_data;
	Vec ot_data;

	std::string pre_ot_filename;

	BaseCot<T> *base_cot = nullptr;
	OTPre<T> *pre_ot = nullptr;
	// ThreadPool *pool = nullptr;
	MpcotReg<T> *mpcot = nullptr;
	LpnF2<T, 10> *lpn_f2 = nullptr;

	void online_sender(block *data, int64_t length);

	void online_recver(block *data, const bool *b, int64_t length);

	void set_param();

	void set_preprocessing_param();

	void extend_initialization();

	void extend(Vec &ot_output, MpcotReg<T> *mpfss, OTPre<T> *preot,
			LpnF2<T, 10> *lpn, Vec &ot_input);

	void extend(Span &ot_output, MpcotReg<T> *mpfss, OTPre<T> *preot,
			LpnF2<T, 10> *lpn, Vec &ot_input);

	void extend_f2k(Span &ot_buffer);

	void extend_f2k();

	int64_t silent_ot_left();

	void write_pre_data128_to_file(void* loc, __uint128_t delta, std::string filename);

	__uint128_t read_pre_data128_from_file(void* pre_loc, std::string filename);
};

#include "emp-ot/ferret/ferret_cot.hpp"
}
#endif// _VOLE_H_
