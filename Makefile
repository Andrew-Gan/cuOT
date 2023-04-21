SRC_DIR := src
INC_DIR := inc

CC := nvcc -std=c++17 -g -lcurand --compiler-options='-std=c++17 -msse2 -msse -march=native -maes -lpthread'
SRC := $(wildcard $(SRC_DIR)/*)
LIB := -lboost_system -lboost_filesystem
INC := -Iinc
OUT := pprf
INPUT_SIZE=20
NUM_TREES=16

# sbatch args
QUEUE=standby
# b node CPU:GPU is 8:1
NUM_CPU=8
NUM_GPU=1

make:
	$(CC) $(SRC) $(LIB) $(INC) -o $(OUT)

run:
	./$(OUT) $(INPUT_SIZE) $(NUM_TREES)

nsys:
	nsys profile --stats=true --output=nsys-stats ./$(OUT) exp $(INPUT_SIZE)

sbatch:
	sbatch -n $(NUM_CPU) -N 1 --gpus-per-node=$(NUM_GPU) -A $(QUEUE) job.sh

plot:
	python plotter.py

clean:
	rm -f $(OUT) slurm*
