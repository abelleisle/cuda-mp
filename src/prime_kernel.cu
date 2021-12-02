/******************************************************************************
* File:             prime_kernel.cu
*
* Author:           Andy Belle-Isle  
* Created:          12/02/21 
* Description:      CUDA kernel code for Miller-Rabin
*****************************************************************************/

#include "bignum.hpp"
#include "bignum_prime.hpp"

/******************************************************************************
*                                 GLOBALS                                     *
******************************************************************************/

const unsigned PRIMES_NUM = 128;
const unsigned BLOCK_SIZE = 32;
const unsigned MR_TRIALS  = BLOCK_SIZE;
const unsigned GRID_SIZE  = PRIMES_NUM;
const unsigned THREADS    = BLOCK_SIZE * GRID_SIZE;

std::array<bignum, GRID_SIZE> local_primes;
std::array<bignum, THREADS>   mr_trials;
std::array<bignum, THREADS>   mp_stacks;

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

    __shared__ bignum *prime       = &primes[bid];
               bignum *random      = &randoms[tid];
               bignum_stack *stack = &stacks[tid];

    /********************
    * REDUCE THE RANDOM *
    ********************/
    bignum *tmp = &stack->data[stack->sp++];
    *tmp = *random; // TMP
    add_i(prime, -4, stack); // Prime - 4
    sync();
    mod_bignum(tmp, prime, random, stack);
    sync();
    add_i(prime, 4, stack); // Prime + 4
    sync();
    stack->sp--;

    /******************************
    * PUT PRIME INTO 2^p * d FORM *
    ******************************/
    if (threadIdx.x == 1) {
        // return d and s
    }
    sync();

}
