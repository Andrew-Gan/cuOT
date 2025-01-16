#include "emp-ot/emp-ot.h"
#include "test/test.h"
#include <sstream>

#include "logger.h"
#include "gpu_tests.h"

using namespace std;

#define SAMPLE_SIZE 1

int port, party;

float test_ferret(int party, NetIO *io, int64_t num_ot, int ngpu) {
	auto start = clock_start();
	FerretCOT<NetIO> * ferretcot = new FerretCOT<NetIO>(party, party, ngpu, &io, false, true, ferret_b13);
	double timeused = time_from(start);

	// RCOT
	// The RCOTs will be generated at internal memory, and copied to user buffer
	start = clock_start();
	int64_t num = 1 << num_ot;
	// test_rcot<FerretCOT<NetIO>>(ferretcot, io, party, num);
	timeused += time_from(start);
	// cout <<"Active FERRET RCOT\t"<<double(num)/test_rcot<FerretCOT<NetIO>>(ferretcot, io, party, num, false)*1e6<<" OTps"<<endl;

	delete ferretcot;
	return timeused/1000;
}

int main(int argc, char** argv) {
	parse_party_and_port(argv, &party, &port);
	NetIO *io = new NetIO(party == ALICE?nullptr:"127.0.0.1",port);

	int64_t length = 24;
	if (argc > 3)
		length = atoi(argv[3]);
	int ngpu = 1;
	if (argc > 4)
		ngpu = atoi(argv[4]);
	if (ngpu > check_cuda(ngpu)) {
		cerr << "Requested too many GPUs" << endl;
	}
	if(length > 30) {
		cout <<"Large test size! comment me if you want to run this size\n";
		exit(1);
	}
	std::stringstream filename;
	filename << "../results/gpu-ferret-";
	if (party == ALICE)
		filename << "send-";
	else
		filename << "recv-";
	filename << length << "-" << ngpu;
	if (party==ALICE) cout << "size: " << length << ", ngpu: " << ngpu << endl;
	if (party==ALICE) cout << "Warming up..." << endl;
	test_ferret(party, io, 10, ngpu);
	// if (party==ALICE) cout << "Benchmarking..." << endl;
	// float duration = 0;
	// Log::open((Role)(party-1), filename.str(), SAMPLE_SIZE);
	// for (int i = 0; i < SAMPLE_SIZE; i++) {
	// 	duration += test_ferret(party, io, length, ngpu);
	// }
	// Log::close((Role)(party-1));
	// printf("%d\truntime\t %.2f ms\n", party, duration / SAMPLE_SIZE);

	delete io;
}