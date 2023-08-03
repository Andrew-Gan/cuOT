CC := nvcc
CUFLG := -g -G -std=c++20
CCFLG := -g -std=c++20 -gdwarf-4 -w
LIB := -lcurand -lcufft
INC := -I0-app -I1-lib -I2-mod -I3-dev
EXE := ot

############################################################

APP_SRC := $(shell find 0-app -name '*.c*')
LIB_SRC := $(shell find 1-lib -name '*.c*')
MOD_SRC := $(shell find 2-mod -name '*.c*')
DEV_SRC := $(shell find 3-dev -name '*.c*')

ALL_HDR := $(shell find 0-app 1-lib 2-mod 3-dev -name '*.h*')

OBJ 	:= obj
APP_OBJ := $(patsubst %.cu, $(OBJ)/app/%.o, $(notdir $(APP_SRC)))

LIB_OBJ := $(patsubst %.c, $(OBJ)/lib/%.o, $(notdir $(LIB_SRC)))
LIB_OBJ := $(patsubst %.cpp, $(OBJ)/lib/%.o, $(LIB_OBJ))

MOD_OBJ := $(patsubst %.cu, $(OBJ)/mod/%.o, $(notdir $(MOD_SRC)))

DEV_OBJ := $(patsubst %.cu, $(OBJ)/dev/%.o, $(notdir $(DEV_SRC)))

FILTER = $(foreach v,$(2),$(if $(findstring $(1),$(v)),$(v)))

############################################################

QUEUE=standby #zghodsi-b
CPU_PER_NODE=16
GPU_PER_NODE=2
NUM_NODE=1
CLUSTER=K

############################################################

.PHONY: all clean

all: $(EXE)

$(EXE): $(APP_OBJ) $(LIB_OBJ) $(MOD_OBJ) $(DEV_OBJ)
	$(CC) $(CUFLG) --compiler-options='$(CCFLG)' $(LIB) $^ -o $(EXE)

$(OBJ)/app/%.o: 0-app/%.cu $(ALL_HDR)
	@mkdir -p $(OBJ)/app
	$(CC) $(CUFLG) --compiler-options='$(CCFLG)' $(LIB) $(INC) -c -o $@ $<

$(OBJ)/lib/%.o:
	@mkdir -p $(OBJ)/lib
	$(CC) $(CUFLG) --compiler-options='$(CCFLG)' $(LIB) $(INC) \
	$(addprefix -I,$(shell find 1-lib -type d -print)) -c -o $@ \
	$(call FILTER,/$(basename $(notdir $@)).,$(LIB_SRC))

$(OBJ)/mod/%.o: 2-mod/%.cu $(ALL_HDR)
	@mkdir -p $(OBJ)/mod
	$(CC) $(CUFLG) --compiler-options='$(CCFLG)' $(LIB) $(INC) -c -o $@ $<

$(OBJ)/dev/%.o: 3-dev/%.cu $(ALL_HDR)
	@mkdir -p $(OBJ)/dev
	$(CC) $(CUFLG) --compiler-options='$(CCFLG)' $(LIB) $(INC) -c -o $@ $<

sbatch:
	sbatch -n $(CPU_PER_NODE) -N $(NUM_NODE) --gpus-per-node=$(GPU_PER_NODE) -A $(QUEUE) --constraint=$(CLUSTER) job.sh

plot:
	python plotter.py

clean:
	rm -rf $(EXE) $(OBJ)/app $(OBJ)/mod $(OBJ)/dev

cleanall:
	rm -rf $(EXE) $(OBJ) prof
