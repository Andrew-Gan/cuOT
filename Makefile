CC := nvcc
CUFLG := -g -G -std=c++20
CCFLG := -g -std=c++20 -gdwarf-4 -w
LIB := -lcurand -lcufft
INC := -I./silent/lib
DIR := gpu-tools
OBJ := gpu_tools.o

############################################################

SRC_FILES := $(shell find $(DIR) -name '*.cu')
HDR_FILES := $(shell find $(DIR) -name '*.h*')
OBJ_FILES := $(patsubst %.cu, %.o, $(SRC_FILES))

FILTER = $(foreach v,$(2),$(if $(findstring $(1),$(v)),$(v)))

############################################################

QUEUE=standby #zghodsi-b
CPU_PER_NODE=16
GPU_PER_NODE=2
NUM_NODE=1
CLUSTER=K

############################################################

.PHONY: all clean

all: $(OBJ)
	cd ferret; python build.py --ot

run: ferret/emp-ot/bin/test_ferret
	rm -f slurm*.out
	sbatch -n 4 -N 1 --gpus-per-node=1 -A standby job-ferret.sh

$(OBJ): $(OBJ_FILES) $(HELPER)
	ld -r -o $(OBJ) $(OBJ_FILES)

$(DIR)/%.o: $(DIR)/%.cu $(HDR_FILES)
	$(CC) $(CUFLG) --compiler-options='$(CCFLG)' $(LIB) $(INC) -c -o $@ $<

clean:
	rm -rf $(DIR)/*.o $(OBJ)
