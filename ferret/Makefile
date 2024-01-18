GPU_TOOLS := ../gpu/gpu.o

############################################################

QUEUE=standby #zghodsi-b
CPU_PER_NODE=32
GPU_PER_NODE=2
CONSTRAINT=A100

############################################################

.PHONY: all clean

ferret: $(GPU_TOOLS)
	python build.py --ot

run: emp-ot/bin/test_ferret
	rm -f slurm*.out report*sqlite report*nsys-rep
	sbatch -n $(CPU_PER_NODE) -N 1 --gpus-per-node=$(GPU_PER_NODE) -A $(QUEUE) --constraint=$(CONSTRAINT) job.sh
