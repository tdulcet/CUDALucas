NAME = CUDALucas
VERSION = 2.03
OptLevel = 3
OUT = $(NAME)

CUC = nvcc
CUFLAGS = -O$(OptLevel) -arch=sm_13 --compiler-options=-Wall
CULIB = /usr/local/cuda/lib64

CC = gcc
CFLAGS = -O$(OptLevel) -Wall

L = -lcufft -lcudart -lm
LDFLAGS = $(CFLAGS) -fPIC -L$(CULIB) $(L)

$(NAME): CUDALucas.o parse.o
	$(CC) $^ $(LDFLAGS) -o $(OUT)
	
CUDALucas.o: CUDALucas.cu parse.h cuda_safecalls.h
	$(CUC) $(CUFLAGS) -c $<

parse.o: parse.c
	$(CC) $(CFLAGS) -c $?

clean: 
	rm -f *.o
cleaner:
	rm -f $(NAME) debug_$(NAME) test_$(NAME)
cleanest: clean cleaner
