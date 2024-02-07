CC := nvcc
CUFLG := -g -G -std=c++20
CCFLG := -g -std=c++20 -gdwarf-4 -w
LIB := -lcurand -lcufft -lcuda
INC := -I../gpu -I./lib
GPU := ../gpu/gpu.o
EXE := ot

############################################################

HDR 	:= $(shell find . lib -name '*.h*')
SRC 	:= $(shell find . -name '*.cu')
OBJ 	:= $(patsubst %.cu, %.o, $(SRC))
LIB_SRC := $(shell find lib -name '*.c*')
LIB_OBJ := $(patsubst %.c, %.o, $(LIB_SRC))
LIB_OBJ := $(patsubst %.cpp, %.o, $(LIB_OBJ))
FILTER = $(foreach v,$(2),$(if $(findstring $(1),$(v)),$(v)))

############################################################

QUEUE=standby #zghodsi-b
CPU_PER_NODE=32
GPU_PER_NODE=2
CONSTRAINT=A100

############################################################

.PHONY: all clean

$(EXE): $(GPU) $(OBJ) $(LIB_OBJ)	
	$(CC) $(CUFLG) --compiler-options='$(CCFLG)' $(LIB) $^ -o $(EXE)

run: $(EXE)
	rm -f slurm*.out report*sqlite report*nsys-rep
	sbatch -n $(CPU_PER_NODE) -N 1 --gpus-per-node=$(GPU_PER_NODE) -A $(QUEUE) --constraint=$(CONSTRAINT) job.sh

%.o: %.cu $(HDR)
	$(CC) $(CUFLG) $(LIB) $(INC) -c -o $@ $<

lib/%.o: lib/%.c*
	$(CC) $(CUFLG) --compiler-options='$(CCFLG)' $(LIB) $(INC) \
	$(addprefix -I,$(shell find lib -type d -print)) -c -o $@ \
	$(call FILTER,/$(basename $(notdir $@)).,$(LIB_SRC))

clean:
	rm -rf *.o ot
