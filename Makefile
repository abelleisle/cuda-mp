###############
#  CPU EXECS  #
###############
CXX = g++
CXX_FLAGS = -g -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2
CXX_SRCS = src/cpu.cpp
CXX_EXEC = cpu

###############
#  GPU EXECS  #
###############
NVCC = nvcc
NVCC_FLAGS = -g -std=c++17
NVCC_SRCS = src/gpu.cu
NVCC_EXEC = gpu

###############
#  LIB EXECS  #
###############
LIB_SRCS = lib/bignum.hpp lib/bignum_prime.hpp lib/compile.hpp
LIB_FLAGS = -Ilib

all: $(CXX_EXEC) $(NVCC_EXEC)

$(CXX_EXEC): $(CXX_SRCS) $(LIB_SRCS)
	$(CXX) $(CXX_FLAGS) $(LIB_FLAGS) $(CXX_SRCS) -o $(CXX_EXEC)

$(NVCC_EXEC): $(NVCC_SRCS) $(LIB_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(LIB_FLAGS) $(NVCC_SRCS) -o $(NVCC_EXEC)
