SRC=pprf.cpp aes.cpp aesni.cpp aesCudaUtils.cpp aesgpu.cu main.cpp
CC=nvcc -g -O3 --compiler-options='-g -msse2 -msse -march=native -maes -lpthread'
LIB=-lboost_system -lboost_filesystem
PLAINTEXT=testData/input.txt
KEY=testData/key.txt
INPUT_SIZE=20

# sbatch args
QUEUE=zghodsi-b
# b node CPU:GPU is 8:1
NUM_CPU=8
NUM_GPU=1

test:
	$(CC) $(SRC) $(LIB) -o aes

enc:
	./aes enc $(PLAINTEXT) $(KEY) $(INPUT_SIZE)

exp:
	./aes exp $(INPUT_SIZE)

nsys:
	nsys profile --stats=true --output=nsys-stats ./aes exp $(INPUT_SIZE)

sbatch:
	sbatch -n $(NUM_CPU) -N 1 --gpus-per-node=$(NUM_GPU) -A $(QUEUE) job.sh

plot:
	python plotter.py

clean:
	rm -f aes slurm*
