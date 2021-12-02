/******************************************************************************
* File:             compile.hpp
*
* Author:           Andy Belle-Isle  
* Created:          11/28/21 
* Description:      Multi-precision arithmetic library compile header. This
*                   header is used to allow for cross-compiler compilation. When
*                   this header is included the same code can be compiled with
*                   both NVCC and GCC even when no NVCC compiler is present.
*****************************************************************************/

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
