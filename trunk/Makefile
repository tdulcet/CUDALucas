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

###############################################################################
# Crazy uses

TARFILES = CUDALucas.cu CUDALucas.ini Makefile Makefile.win parse.c parse.h \
	cuda_safecalls.h timeval.c
TARBALL = $(NAME)-$(VERSION).tar.bz2
TAR = tar

tar: $(TARFILES)
	$(TAR) cj $^ --file $(TARBALL)

debug: CFLAGS += -DEBUG	-g
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
	@echo "\"make tar\"       creates a compressed archive with the \
	source files"
	@echo "\"make debug\"     creates a debug build"
	@echo "\"make test\"      creates an experimental build"
	@echo "\"make help\"      prints this message\n"
