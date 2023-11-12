CC := nvcc
CUFLG := -g -G -std=c++20
CCFLG := -g -std=c++20 -gdwarf-4 -w
LIB := -lcurand -lcufft
INC := -I./silent/lib
DIR := gpu-tools

############################################################

SRC := $(shell find $(DIR) -name '*.cu')
HDR := $(shell find $(DIR) -name '*.h*')
OBJ := $(patsubst %.cu, %.o, $(SRC))

FILTER = $(foreach v,$(2),$(if $(findstring $(1),$(v)),$(v)))

############################################################

QUEUE=standby #zghodsi-b
CPU_PER_NODE=16
GPU_PER_NODE=2
NUM_NODE=1
CLUSTER=K

############################################################

.PHONY: all clean

ferret: gpu_tools.o
	cd ferret; \
	python build.py --ot

gpu_tools.o: $(OBJ) $(HELPER)
	ld -r -o gpu_tools.o $(OBJ)

$(DIR)/%.o: $(DIR)/%.cu $(HDR)
	$(CC) $(CUFLG) --compiler-options='$(CCFLG)' $(LIB) $(INC) -c -o $@ $<

clean:
	rm -rf $(DIR)/*.o
