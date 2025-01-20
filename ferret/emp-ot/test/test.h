#include <emp-tool/emp-tool.h>
#include "emp-ot/emp-ot.h"
#include <iostream>
using namespace emp;

template <typename T>
double test_ot(T * ot, NetIO *io, int party, int64_t length) {
	block *b0 = new block[length], *b1 = new block[length],
	*r = new block[length];
	PRG prg(fix_key);
	prg.random_block(b0, length);
	prg.random_block(b1, length);
	bool *b = new bool[length];
	PRG prg2;
	prg2.random_bool(b, length);

	auto start = clock_start();
	if (party == ALICE) {
		ot->send(b0, b1, length);
	} else {
		ot->recv(r, b, length);
	}
	io->flush();
	long long t = time_from(start);
	if (party == BOB) {
		for (int64_t i = 0; i < length; ++i) {
			if (b[i]){ if(!cmpBlock(&r[i], &b1[i], 1)) {
				std::cout <<i<<"\n";
				error("wrong!\n");
			}}
			else { if(!cmpBlock(&r[i], &b0[i], 1)) {
				std::cout <<i<<"\n";
				error("wrong!\n");
			}}
		}
	}
	std::cout << "Tests passed.\t";
	delete[] b0;
	delete[] b1;
	delete[] r;
	delete[] b;
	return t;
}


template <typename T>
double test_cot(T * ot, NetIO *io, int party, int64_t length) {
	block *b0 = new block[length], *r = new block[length];
	bool *b = new bool[length];
	block delta;
	PRG prg;
	prg.random_block(&delta, 1);
	prg.random_bool(b, length);

	io->sync();
	auto start = clock_start();
	if (party == ALICE) {
		ot->send_cot(b0, length);
		delta = ot->Delta;
	} else {
		ot->recv_cot(r, b, length);
	}
	io->flush();
	long long t = time_from(start);
	if (party == ALICE) {
		io->send_block(&delta, 1);
		io->send_block(b0, length);
	}
	else if (party == BOB) {
		io->recv_block(&delta, 1);
		io->recv_block(b0, length);
		for (int64_t i = 0; i < length; ++i) {
			block b1 = b0[i] ^ delta;
			if (b[i]) {
				if (!cmpBlock(&r[i], &b1, 1))
					error("COT failed!");
			} else {
				if (!cmpBlock(&r[i], &b0[i], 1))
					error("COT failed!");
			}
		}
	}
	std::cout << "Tests passed.\t";
	io->flush();
	delete[] b0;
	delete[] r;
	delete[] b;
	return t;
}

template <typename T>
double test_rot(T* ot, NetIO *io, int party, int64_t length) {
	block *b0 = new block[length], *r = new block[length];
	block *b1 = new block[length];
	bool *b = new bool[length];
	PRG prg;
	prg.random_bool(b, length);

	io->sync();
	auto start = clock_start();
	if (party == ALICE) {
		ot->send_rot(b0, b1, length);
	} else {
		ot->recv_rot(r, b, length);
	}
	io->flush();
	long long t = time_from(start);
	if (party == ALICE) {
		io->send_block(b0, length);
		io->send_block(b1, length);
	} else if (party == BOB) {
		io->recv_block(b0, length);
		io->recv_block(b1, length);
		for (int64_t i = 0; i < length; ++i) {
			if (b[i])
				assert(cmpBlock(&r[i], &b1[i], 1));
			else
				assert(cmpBlock(&r[i], &b0[i], 1));
		}
	}
	std::cout << "Tests passed.\t";
	io->flush();
	delete[] b0;
	delete[] b1;
	delete[] r;
	delete[] b;
	return t;
}

template <typename T>
double test_rcot(T* ot, NetIO *io, int party, int64_t length) {
	// Mat b({(uint64_t)length});
	PRG prg;
	io->sync();
	auto start = clock_start();
	int64_t mem_size;
	mem_size = length;
	// The RCOTs will be generated in the internal buffer
	// then be copied to the user buffer
	Mat b({length});
	ot->rcot(b, length);
	long long t = time_from(start);
	io->sync();

	block *b_h = new block[length];
	b.write_to_cpu(b_h, b.size_bytes());

	if (party == ALICE) {
		io->send_block(&ot->Delta, 1);
		io->send_block(b_h, mem_size);
	}
	else if (party == BOB) {
		block ch[2];
		ch[0] = zero_block;
		block *b0 = new block[mem_size];
		io->recv_block(ch+1, 1);
		io->recv_block(b0, mem_size);
		for (int64_t i = 0; i < mem_size; ++i) {
			b_h[i] = b_h[i] ^ ch[getLSB(b_h[i])];
		}
		if (!cmpBlock(b_h, b0, mem_size)) {
			std::cerr << "RCOT failed" << std::endl;
			return 0;
		}
		std::cout << "Tests passed.\t";
		delete[] b0;
	}
	
	delete[] b_h;
	return t;
}