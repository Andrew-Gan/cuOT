GPU_TOOLS := ../gpu/gpu.o

############################################################

QUEUE=standby #zghodsi-b
CPU_PER_NODE=8
GPU_PER_NODE=2
CLUSTER=K

############################################################

.PHONY: all clean

ferret: $(GPU_TOOLS)
	python build.py --ot

run: emp-ot/bin/test_ferret
	rm -f slurm*.out report*sqlite report*nsys-rep
	sbatch -n $(CPU_PER_NODE) -N 1 --gpus-per-node=$(GPU_PER_NODE) -A $(QUEUE) --constraint=$(CLUSTER) job.sh
