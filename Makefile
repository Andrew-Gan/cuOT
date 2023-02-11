SRC=main.c pprf.c aes.c aesni.c aesCudaUtils.cpp aesgpu.cu
CC=nvcc -g -O3 --compiler-options='-g -msse2 -msse -march=native -maes -lpthread'
LIB=-lboost_system -lboost_filesystem
PLAINTEXT=testData/input.txt
KEY=testData/key.txt
INPUT_SIZE=8
NUM_THREADS=1

test:
	$(CC) $(SRC) $(LIB) -o aes

enc:
	./aes enc $(PLAINTEXT) $(KEY) $(INPUT_SIZE) $(NUM_THREADS)

exp:
	./aes exp $(INPUT_SIZE) $(NUM_THREADS)

sbatch:
	sbatch --nodes=1 --gpus-per-node=1 -A standby job.sh

plot:
	python plotter.py

clean:
	rm -f aes slurm*
