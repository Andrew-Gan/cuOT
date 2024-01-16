CC := nvcc
FLG := -g -G -std=c++20
LIB := -lcurand -lcufft -lcuda
INC := -I./silent/lib
FINAL := gpu.o

############################################################

SRC := $(shell find . -name '*.cu')
HDR := $(shell find . -name '*.h*')
OBJ := $(patsubst %.cu, %.o, $(SRC))

############################################################

.PHONY: all clean

$(FINAL): $(OBJ) $(HDR)
	ld -r -o $(FINAL) $(OBJ)

%.o: %.cu $(HDR)
	$(CC) $(FLG) $(LIB) $(INC) -c -o $@ $<

clean:
	rm -rf *.o
