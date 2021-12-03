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

const unsigned PRIMES_NUM = 128;
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

    if (bid >= PRIMES_NUM)
        return;
    if (tid >= THREADS)
        return;

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
        if (even_bignum(prime))
            add_i(prime, 1, stack);
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

/*
function bytesToSize(bytes) {
   var sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
   if (bytes == 0) return '0 Byte';
   var i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
   return Math.round(bytes / Math.pow(1024, i), 2) + ' ' + sizes[i];
}
*/

std::string sz(size_t bytes)
{
    std::string ret;
    std::string sizes[4] = {" Bytes", " KB", " MB", " GB"};
    if (bytes == 0) {
        ret = "0 Bytes";
    } else {
        int i = floor(log2(bytes)/10.0f);
        if (i > 4) {
            ret = std::to_string(bytes) + sizes[0];
        } else {
            int b = ceil(bytes / pow(1024, i));
            ret = std::to_string(b) + sizes[i];
        }
    }

    return ret;
}

// TODO: error handle
bool find_primes(void)
{
    /*********************
    * MEMORY USAGE STATS *
    *********************/

    size_t bignum_sz = sizeof(bignum);
    size_t bigstack_sz = sizeof(bignum_stack);

    std::cout << "Memory Usage statistics: " << std::endl;
    std::cout << " - Per Thread: " << sz(bignum_sz + bignum_sz + bigstack_sz) << std::endl;
    std::cout << "   - MP Stack: " << sz(bigstack_sz) << std::endl;
    std::cout << "   - Prime:    " << sz(bignum_sz) << std::endl;
    std::cout << "   - Random:   " << sz(bignum_sz) << std::endl;
    std::cout << " - Per Block:  " << sz(bignum_sz * BLOCK_SIZE * 2 
                                    + bigstack_sz * BLOCK_SIZE) << std::endl;
    std::cout << "   - MP Stack: " << sz(bigstack_sz * BLOCK_SIZE) << std::endl;
    std::cout << "   - Prime:    " << sz(bignum_sz * BLOCK_SIZE) << std::endl;
    std::cout << "   - Random:   " << sz(bignum_sz * BLOCK_SIZE) << std::endl;
    std::cout << " - Per GPU:    " << sz(bignum_sz * GRID_SIZE * 2 + 
                                    + bigstack_sz * THREADS) << std::endl;
    std::cout << "   - MP Stack: " << sz(bigstack_sz * THREADS) << std::endl;
    std::cout << "   - Prime:    " << sz(bignum_sz * GRID_SIZE) << std::endl;
    std::cout << "   - Random:   " << sz(bignum_sz * GRID_SIZE) << std::endl;

    /*************************
    * FILL ARRAYS FOR PRIMES *
    *************************/

    /* Fill our random prime searches */
    std::cout << "Generating prime attempts..." << std::endl;
    for (auto &r : local_primes)
        rand_digits_bignum(&r, 155);

    /* Fill our random trials */
    std::cout << "Generating prime trials..." << std::endl;
    for (auto &t : mr_trials)
        rand_digits_bignum(&t, 155);

    /***********************
    * ALLOCATE CUDA MEMORY *
    ***********************/

    bignum *cuda_primes;
    bignum *cuda_trials;
    bignum_stack *cuda_stacks;

    cudaError_t error;

    /* Prime number allocation */
    error = cudaMalloc(&cuda_primes, sizeof(bignum) * PRIMES_NUM); 
    if (error != cudaSuccess) {
        std::cerr << "Unable to allocate CUDA memory for prime storage" << std::endl;
        abort();
    }
    /* Prime trial allocation */
    error = cudaMalloc(&cuda_trials, sizeof(bignum) * THREADS);
    if (error != cudaSuccess) {
        std::cerr << "Unable to allocate CUDA memory for prime trials" << std::endl;
        cudaFree(cuda_primes); abort();
    }
    /* Thread stack allocations */
    error = cudaMalloc(&cuda_stacks, sizeof(bignum_stack) * THREADS);
    if (error != cudaSuccess) {
        std::cerr << "Unable to allocate CUDA memory for thread stacks" << std::endl;
        cudaFree(cuda_primes); cudaFree(cuda_trials); abort();
    }

    error = cudaMemcpy(cuda_primes, local_primes.data(),
                       sizeof(bignum) * PRIMES_NUM, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Unable to copy prime numbers to CUDA memory" << std::endl;
        cudaFree(cuda_primes); cudaFree(cuda_trials); cudaFree(cuda_stacks); abort();
    }

    error = cudaMemcpy(cuda_trials, mr_trials.data(),
                       sizeof(bignum) * THREADS, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Unable to copy trials numbers to CUDA memory" << std::endl;
        cudaFree(cuda_primes); cudaFree(cuda_trials); cudaFree(cuda_stacks); abort();
    }

    /*****************
    * RUN THE KERNEL *
    *****************/

    std::cout << "Running kernel" << std::endl;
    cudaDeviceSynchronize();
    MR_CUDA<<<GRID_SIZE, BLOCK_SIZE>>>(cuda_primes, cuda_trials, cuda_stacks);
    cudaDeviceSynchronize();

    /***********************
    * COPY AND FREE MEMORY *
    ***********************/

    cudaMemcpy(local_primes.data(), cuda_primes,
               sizeof(bignum) * PRIMES_NUM, cudaMemcpyDeviceToHost);

    cudaFree(cuda_primes);
    cudaFree(cuda_trials);
    cudaFree(cuda_stacks);

    /*****************
    * OUTPUT RESULTS *
    *****************/

    bignum zero;
    int primecount = 0;
    int_to_bignum(0, &zero);

    for (auto &p : local_primes) {
        if (compare_bignum(&zero, &p) != 0) {
            print_bignum(&p);
            primecount++;
        }
    }

    std::cout << "Found " << primecount << " primes out of " << PRIMES_NUM 
              << " numbers tested." << std::endl;

    return true;
}
