CC := nvcc -g -G -std=c++17 -lcurand --compiler-options='-g'
LIB :=
INC := -I0-app -I1-module -I2-device
EXE := ot
INPUT_SIZE=20
NUM_TREES=16

############################################################

APP_SRC := $(wildcard 0-app/*.cu)
MOD_SRC := $(wildcard 1-module/*.cu)
DEV_SRC := $(wildcard 2-device/*.cu)

OBJ 	:= obj
APP_OBJ := $(patsubst 0-app/%.cu, $(OBJ)/app/%.o, $(APP_SRC))
MOD_OBJ := $(patsubst 1-module/%.cu, $(OBJ)/module/%.o, $(MOD_SRC))
DEV_OBJ := $(patsubst 2-device/%.cu, $(OBJ)/device/%.o, $(DEV_SRC))

############################################################

QUEUE=standby
NUM_CPU=8
NUM_GPU=1

############################################################

.PHONY: all clean

all: $(EXE)

$(EXE): $(APP_OBJ) $(MOD_OBJ) $(DEV_OBJ)
	$(CC) $^ -o $(EXE)

$(OBJ)/app/%.o: 0-app/%.cu | $(OBJ)
	$(CC) $(INC) -c -o $@ $<

$(OBJ)/module/%.o: 1-module/%.cu | $(OBJ)
	$(CC) $(INC) -c -o $@ $<

$(OBJ)/device/%.o: 2-device/%.cu | $(OBJ)
	$(CC) $(INC) -c -o $@ $<

$(OBJ):
	mkdir $@ $@/app $@/module $@/device

nsys:
	nsys profile --stats=true --output=nsys-stats ./$(EXE) exp $(INPUT_SIZE)

sbatch:
	sbatch -n $(NUM_CPU) -N 1 --gpus-per-node=$(NUM_GPU) -A $(QUEUE) job.sh

plot:
	python plotter.py

clean:
	rm -f $(EXE) obj/*/*.o
