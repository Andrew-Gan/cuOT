CC := nvcc -g -G -std=c++17 --compiler-options='-g'
LIB := -lcurand
INC := -I0-app -I1-module -I2-device
EXE := ot

############################################################

APP_SRC := $(shell find 0-app -name '*.cu')
MOD_SRC := $(shell find 1-module -name '*.cu')
DEV_SRC := $(shell find 2-device -name '*.cu')

OBJ 	:= obj
APP_OBJ := $(patsubst 0-app/%.cu, $(OBJ)/app/%.o, $(APP_SRC))
MOD_OBJ := $(patsubst 1-module/%.cu, $(OBJ)/module/%.o, $(MOD_SRC))
DEV_OBJ := $(patsubst 2-device/%.cu, $(OBJ)/device/%.o, $(DEV_SRC))

############################################################

QUEUE=standby #zghodsi-b
NUM_CPU=16
NUM_GPU=2

############################################################

.PHONY: all clean

all: $(EXE)

$(EXE): $(APP_OBJ) $(MOD_OBJ) $(DEV_OBJ)
	$(CC) $(LIB) $^ -o $(EXE)

$(OBJ)/app/%.o: 0-app/%.cu | $(OBJ)
	$(CC) $(LIB) $(INC) -c -o $@ $<

$(OBJ)/module/%.o: 1-module/%.cu | $(OBJ)
	$(CC) $(LIB) $(INC) -c -o $@ $<

$(OBJ)/device/%.o: 2-device/%.cu | $(OBJ)
	$(CC) $(LIB) $(INC) -c -o $@ $<

$(OBJ):
	mkdir $@ $@/app $@/module $@/device $@/module/blake2 $@/module/blake2/c $@/module/blake2/sse

sbatch:
	sbatch -n $(NUM_CPU) -N 1 --gpus-per-node=$(NUM_GPU) -A $(QUEUE) job.sh

plot:
	python plotter.py

clean:
	rm -f $(EXE)
	find . -name '*.o' -type f -delete
