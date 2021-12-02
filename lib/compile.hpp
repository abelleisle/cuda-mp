#pragma once

#if defined(__CUDACC__)
#warning using NVCC
#include "curand.h"
#include "curand_kernel.h"
#define FNC_H __host__
#define FNC_D __device__ __inline__
#define FNC_DH __device__ __host__ __inline__
#define FNC_DH __device__ __host__ __inline__
#define sync() __syncthreads()

#else
#define FNC_D
#define FNC_H
#define FNC_DH
#define max(a, b) std::max(a,b)
#define sync()

#endif
