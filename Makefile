###############
#  CPU EXECS  #
###############
CXX = g++
CXX_FLAGS = -g -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2
CXX_SRCS = src/cpu.cpp
CXX_EXEC = cpu

###############
#  LIB EXECS  #
###############
LIB_SRCS = lib/bignum.hpp lib/bignum_prime.hpp lib/compile.hpp
LIB_FLAGS = -Ilib

all: $(CXX_EXEC)

$(CXX_EXEC): $(CXX_SRCS) $(LIB_SRCS)
	$(CXX) $(CXX_FLAGS) $(LIB_FLAGS) $(CXX_SRCS) -o $(CXX_EXEC)
