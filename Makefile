SRC=main.c pprf.c aes.c aesni.c aesCudaUtils.cpp aesgpu.cu
CC=nvcc -g -O3 --compiler-options='-g -msse2 -msse -march=native -maes -lpthread'
LIB=-lboost_system -lboost_filesystem
PLAINTEXT=testData/input.txt
KEY=testData/key.txt
INPUT_SIZE=8

# sbatch args
QUEUE=zghodsi-b
NUM_CPU=8
NUM_GPU=1

test:
	$(CC) $(SRC) $(LIB) -o aes

enc:
	./aes enc $(PLAINTEXT) $(KEY) $(INPUT_SIZE) $(NUM_THREADS)

exp:
	./aes exp $(INPUT_SIZE)

sbatch:
	sbatch -n $(NUM_CPU) -N 1 --gpus-per-node=$(NUM_GPU) -A $(QUEUE) job.sh

plot:
	python plotter.py

clean:
	rm -f aes slurm*
