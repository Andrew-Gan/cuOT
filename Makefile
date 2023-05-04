CC := nvcc -std=c++17 -lcurand
SRC := $(wildcard */*.cu)
LIB :=
INC := -I0_app_level -I1_module_level -I2_gpu_level
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

nsys:
	nsys profile --stats=true --output=nsys-stats ./$(OUT) exp $(INPUT_SIZE)

sbatch:
	sbatch -n $(NUM_CPU) -N 1 --gpus-per-node=$(NUM_GPU) -A $(QUEUE) job.sh

plot:
	python plotter.py

clean:
	rm -f $(OUT)
