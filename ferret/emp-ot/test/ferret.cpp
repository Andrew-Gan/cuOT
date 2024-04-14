#include "emp-ot/emp-ot.h"
#include "test/test.h"
#include <sstream>
using namespace std;

#define SAMPLE_SIZE 8

int port, party;

void test_ferret(int party, NetIO *io, int64_t num_ot, int ngpu) {
	// auto start = clock_start();
	FerretCOT<NetIO> * ferretcot = new FerretCOT<NetIO>(party, ngpu, io, false, true, ferret_b13);
	// double timeused = time_from(start);
	// std::cout << party << "\tsetup\t" << timeused/1000 << "ms" << std::endl;

	// RCOT
	// The RCOTs will be generated at internal memory, and copied to user buffer
	int64_t num = 1 << num_ot;
	test_rcot<FerretCOT<NetIO>>(ferretcot, io, party, num, false);
	// cout <<"Active FERRET RCOT\t"<<double(num)/test_rcot<FerretCOT<NetIO>>(ferretcot, io, party, num, false)*1e6<<" OTps"<<endl;

	// RCOT inplace
	// The RCOTs will be generated at user buffer
	// Get the buffer size needed by calling byte_memory_need_inplace()
	// uint64_t batch_size = ferretcot->ot_limit;
	// cout <<"Active FERRET RCOT inplace\t"<<double(batch_size)/test_rcot<FerretCOT<NetIO>>(ferretcot, ios[0], party, batch_size, true)*1e6<<" OTps"<<endl;
	delete ferretcot;
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
	filename << length << "-" << ngpu << ".txt";
	if (party==ALICE) cout << "size: " << length << ", ngpu: " << ngpu << endl;
	if (party==ALICE) cout << "Warming up..." << endl;
	test_ferret(party, io, 10, ngpu); // warmup
	if (party==ALICE) cout << "Benchmarking..." << endl;
	Log::open((Role)(party-1), filename.str(), true, SAMPLE_SIZE);
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		test_ferret(party, io, length, ngpu);
	}
	Log::close((Role)(party-1));

	delete io;
}
