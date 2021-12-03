/******************************************************************************
* File:             prime_kernel.cu
*
* Author:           Andy Belle-Isle  
* Created:          12/02/21 
* Description:      CUDA kernel code for Miller-Rabin
*****************************************************************************/

#include "bignum.hpp"
#include "bignum_prime.hpp"

#include <array>

/******************************************************************************
*                                 GLOBALS                                     *
******************************************************************************/

const unsigned PRIMES_NUM = 12800;
const unsigned BLOCK_SIZE = 32;
const unsigned MR_TRIALS  = BLOCK_SIZE;
const unsigned GRID_SIZE  = PRIMES_NUM;
const unsigned THREADS    = BLOCK_SIZE * GRID_SIZE;

std::array<bignum, GRID_SIZE> local_primes;
std::array<bignum, THREADS>   mr_trials;
//std::array<bignum, THREADS>   mp_stacks;

/******************************************************************************
*                                 KERNELS                                     *
******************************************************************************/

/** @brief Miller-Rabin CUDA code to test primality on GPU.
 * 
 * @param[primes]  - The list of prime numbers to test. If a number of prime it
 *                   will remain in the array, otherwise 0 should be written to
 *                   that index.
 * @param[randoms] - The random numbers used during Miller-Rabin trials. This
 *                   array is the (# MR trials) * (# primes). These prime are
 *                   NOT reduced and must be reduced to the range of 2 to prime
 *                   - 2;
 * @param[stacks]  - The arithmetic stacks used for MP workloads. This is the
 *                   size of the number of threads per block.
 *
 * @note This should be run with the same number of threads per block as MR
 *       trials.
 */
__global__ void MR_CUDA(bignum *primes, bignum *randoms, bignum_stack *stacks)
{
    auto bid = blockIdx.x;
    auto tid = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ bignum *prime;
               bignum *random      = &randoms[tid];
               bignum_stack *stack = &stacks[tid];

    __shared__ int preprime;
    __shared__ int isnotprime;

    stack->sp = 0;

    int c = 0; // Factor of d into 2
    bignum *d = &stack->data[stack->sp++]; // prime factor of 2

    /******************************
    * PUT PRIME INTO 2^p * d FORM *
    ******************************/
    if (threadIdx.x == 0) {
        preprime = 0;
        isnotprime = 0;

        prime = &primes[bid];
        preprime = mr_bignum_factor(prime, d, &c, stack);

        if (preprime < 0) {
            int_to_bignum(0, prime);
        }
    }
    sync();
    if (preprime < 0) { // This is a prime, return it
        stack->sp -= 1;
        return;
    }

    /********************
    * REDUCE THE RANDOM *
    ********************/
    mr_bignum_treatrand(prime, random, stack);
    sync();

    /**********************************
    * PERFORM THE SINGLE FERMAT TRIAL *
    **********************************/
    int result = mr_bignum_innerloop(prime, d, c, random, stack);
    if (result == -1)
        atomicAdd(&isnotprime, 1);
    sync();

    stack->sp -= 1;

    if (threadIdx.x == 0)
        if (isnotprime > 0)
            int_to_bignum(0, prime);
    sync();
}

/******************************************************************************
*                                FUNCTIONS                                    *
******************************************************************************/

// TODO: error handle
bool find_primes(void)
{
    /* Fill our random prime searches */
    std::cout << "Generating prime attempts..." << std::endl;
    for (auto &r : local_primes)
        rand_digits_bignum(&r, 20);

    /* Fill our random trials */
    std::cout << "Generating prime trials..." << std::endl;
    for (auto &t : mr_trials)
        rand_digits_bignum(&t, 20);

    bignum *cuda_primes;
    bignum *cuda_trials;
    bignum_stack *cuda_stacks;

    cudaMalloc(&cuda_primes, sizeof(bignum) * PRIMES_NUM); 
    cudaMalloc(&cuda_trials, sizeof(bignum) * THREADS);
    cudaMalloc(&cuda_stacks, sizeof(bignum_stack) * THREADS);

    cudaMemcpy(cuda_primes, local_primes.data(),
               sizeof(bignum) * PRIMES_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_trials, mr_trials.data(),
               sizeof(bignum) * THREADS, cudaMemcpyHostToDevice);
    
    std::cout << "Running kernel" << std::endl;
    MR_CUDA<<<GRID_SIZE, BLOCK_SIZE>>>(cuda_primes, cuda_trials, cuda_stacks);
    cudaDeviceSynchronize();

    cudaMemcpy(local_primes.data(), cuda_primes,
               sizeof(bignum) * PRIMES_NUM, cudaMemcpyDeviceToHost);

    cudaFree(cuda_primes);
    cudaFree(cuda_trials);
    cudaFree(cuda_stacks);

    bignum zero;
    int_to_bignum(0, &zero);

    for (auto &p : local_primes) {
        if (compare_bignum(&zero, &p) != 0)
            print_bignum(&p);
    }

    return true;
}
