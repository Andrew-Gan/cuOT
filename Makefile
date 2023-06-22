CC := nvcc -g -G -std=c++20 --compiler-options='-g -std=c++20 -gdwarf-4'
LIB := -lcurand
INC := -I0-app -I2-mod -I3-dev $(addprefix -I,$(shell find 1-lib -type d -print))
EXE := ot

############################################################

APP_SRC := $(shell find 0-app -name '*.c*')
LIB_SRC := $(shell find 1-lib -name '*.c*')
MOD_SRC := $(shell find 2-mod -name '*.c*')
DEV_SRC := $(shell find 3-dev -name '*.c*')

OBJ 	:= obj
APP_OBJ := $(patsubst %.cu, $(OBJ)/app/%.o, $(notdir $(APP_SRC)))

LIB_OBJ := $(patsubst %.cu, $(OBJ)/lib/%.o, $(notdir $(LIB_SRC)))
LIB_OBJ := $(patsubst %.cpp, $(OBJ)/lib/%.o, $(LIB_OBJ))
LIB_OBJ := $(patsubst %.c, $(OBJ)/lib/%.o, $(LIB_OBJ))

MOD_OBJ := $(patsubst %.cu, $(OBJ)/mod/%.o, $(notdir $(MOD_SRC)))

DEV_OBJ := $(patsubst %.cu, $(OBJ)/dev/%.o, $(notdir $(DEV_SRC)))

FILTER = $(foreach v,$(2),$(if $(findstring $(1),$(v)),$(v)))

############################################################

QUEUE=standby #zghodsi-b
NUM_CPU=16
NUM_GPU=2

############################################################

.PHONY: all clean

all: $(EXE)

$(EXE): $(APP_OBJ) $(LIB_OBJ) $(MOD_OBJ) $(DEV_OBJ)
	$(CC) $(LIB) $^ -o $(EXE)

$(OBJ)/app/%.o: $(OBJ)
	$(CC) $(LIB) $(INC) -c -o $@ $(call FILTER,/$(basename $(notdir $@)).,$(APP_SRC))

$(OBJ)/lib/%.o: $(OBJ)
	$(CC) $(LIB) $(INC) -c -o $@ $(call FILTER,/$(basename $(notdir $@)).,$(LIB_SRC))

$(OBJ)/mod/%.o: $(OBJ)
	$(CC) $(LIB) $(INC) -c -o $@ $(call FILTER,/$(basename $(notdir $@)).,$(MOD_SRC))

$(OBJ)/dev/%.o: $(OBJ)
	$(CC) $(LIB) $(INC) -c -o $@ $(call FILTER,/$(basename $(notdir $@)).,$(DEV_SRC))

$(OBJ):
	mkdir $@ $@/app $@/lib $@/mod $@/dev

sbatch:
	sbatch -n $(NUM_CPU) -N 1 --gpus-per-node=$(NUM_GPU) -A $(QUEUE) job.sh

plot:
	python plotter.py

clean:
	rm -rf $(EXE) $(OBJ)
