NAME = CUDALucas
VERSION = 2.06
OptLevel = 1

OUT = $(NAME)

# Make sure this points to your cuda installation
CUDA = /usr/local/cuda

CUC = $(CUDA)/bin/nvcc
CULIB = $(CUDA)/lib64
CUINC = $(CUDA)/include

# In CUFLAGSS, make an entry: --generate-code arch=compute_XY,code=sm_XY
# for any compute capability you want to support. Possibilities are
# For cuda 4.2, XY = 13, 20, or 30
# For cuda 5.0, XY = 13, 20, 30, or 35
# For cuda 5.5, XY = 13, 20, 30, or 35
# For cuda 6.0, XY = 13, 20, 30, 35, or 50
# For cuda 6.5, XY = 13, 20, 30, 35, or 50
# For cuda 7.0, XY = 20, 30, 35, 50, or 52

CUFLAGS = -O$(OptLevel)  --generate-code arch=compute_35,code=sm_35 --compiler-options=-Wall -I$(CUINC)
CC = gcc
CFLAGS = -O$(OptLevel) -Wall

L = -lcufft -lcudart -lm #-lnvidia-ml
LDFLAGS = $(CFLAGS) -fPIC -L$(CULIB) $(L)

$(NAME): CUDALucas.o parse.o
	$(CC) $^ $(LDFLAGS) -o $(OUT)

CUDALucas.o: CUDALucas.cu parse.h cuda_safecalls.h
	$(CUC) $(CUFLAGS) -c $<

parse.o: parse.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o
cleaner:
	rm -f $(NAME) debug_$(NAME) test_$(NAME)
cleanest: clean cleaner

debug: CFLAGS += -DEBUG -g
debug: CUFLAGS += -DEBUG -g
debug: OptLevel = 0
debug: OUT = debug_$(NAME)
debug: $(NAME)

test: CFLAGS += -DTEST
test: CUFLAGS += -DTEST
test: OUT = test_$(NAME)
test: $(NAME)

help:
	@echo "\n\"make\"           builds CUDALucas"
	@echo "\"make clean\"     removes object files"
	@echo "\"make cleaner\"   removes executables"
	@echo "\"make cleanest\"  does both clean and cleaner"
	@echo "\"make debug\"     creates a debug build"
	@echo "\"make test\"      creates an experimental build"
	@echo "\"make help\"      prints this message\n"
