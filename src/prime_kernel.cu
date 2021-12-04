/******************************************************************************
* File:             prime_kernel.cu
*
* Author:           Andy Belle-Isle  
* Created:          12/02/21 
* Description:      CUDA kernel code for Miller-Rabin
*****************************************************************************/

#include "bignum.hpp"
#include "bignum_types.hpp"
#include "bignum_prime.hpp"

#include <array>

/******************************************************************************
*                                 GLOBALS                                     *
******************************************************************************/

#define DIGIT_WIDTH 100
#define STACK_DEPTH 15

const unsigned RAND_DIGITS = 20;
const unsigned PRIMES_NUM = 25600;
const unsigned BLOCK_SIZE = 32;
const unsigned MR_TRIALS  = BLOCK_SIZE;
const unsigned GRID_SIZE  = PRIMES_NUM;
const unsigned THREADS    = BLOCK_SIZE * GRID_SIZE;

std::array<cmp::bigint<DIGIT_WIDTH>, GRID_SIZE> local_primes;
std::array<cmp::bigint<DIGIT_WIDTH>, THREADS>   mr_trials;
cmp::env<STACK_DEPTH, DIGIT_WIDTH> local_stack;

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
template<size_t D, size_t N>
__global__ void MR_CUDA(cmp::bigint<N> *primes,
                        cmp::bigint<N> *randoms,
                        cmp::env<D, N> *stacks)
{
    auto bid = blockIdx.x;
    auto tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (bid >= PRIMES_NUM)
        return;
    if (tid >= THREADS)
        return;

    __shared__ cmp::bigint<N> *prime;
               cmp::bigint<N> *random = &randoms[tid];
               cmp::env<D, N> *stack  = &stacks[tid];

    __shared__ int preprime;
    __shared__ int isnotprime;

    stack->sp = 0;

    int c = 0; // Factor of d into 2
    cmp::bigint<N> *d = &stack->data[stack->sp++]; // prime factor of 2

    /******************************
    * PUT PRIME INTO 2^p * d FORM *
    ******************************/
    if (threadIdx.x == 0) {
        preprime = 0;
        isnotprime = 0;

        prime = &primes[bid];
        if (cmp::even(prime))
            cmp::add_i(prime, 1, stack);
        preprime = cmp::mr_factor(prime, d, &c, stack);

        if (preprime < 0) {
            cmp::to_int(0, prime);
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
    cmp::mr_treatrand(prime, random, stack);
    sync();

    /**********************************
    * PERFORM THE SINGLE FERMAT TRIAL *
    **********************************/
    int result = cmp::mr_innerloop(prime, d, c, random, stack);
    if (result == -1)
        atomicAdd(&isnotprime, 1);
    sync();

    stack->sp -= 1;

    if (threadIdx.x == 0)
        if (isnotprime > 0)
            cmp::to_int(0, prime);
    sync();
}

/******************************************************************************
*                                FUNCTIONS                                    *
******************************************************************************/

/** @brief Given a number of bytes, print it all pretty like
 * @param[bytes] - Number of bytes
 * @return[std::string] - Bytes, but prettified
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

/** @brief Find primes using the CUDA GPU.
 * The parameters can be changed at the top of this file.
 */
bool find_primes(void)
{
    /*********************
    * MEMORY USAGE STATS *
    *********************/

    const size_t bignum_sz = sizeof(cmp::bigint<DIGIT_WIDTH>);
    const size_t bigstack_sz = sizeof(cmp::env<STACK_DEPTH, DIGIT_WIDTH>);

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
        cmp::rand_digits(&r, RAND_DIGITS);

    /* Fill our random trials */
    std::cout << "Generating prime trials..." << std::endl;
    for (auto &t : mr_trials)
        cmp::rand_digits(&t, RAND_DIGITS);

    /***********************
    * ALLOCATE CUDA MEMORY *
    ***********************/

    cmp::bigint<DIGIT_WIDTH> *cuda_primes;
    cmp::bigint<DIGIT_WIDTH> *cuda_trials;
    cmp::env<STACK_DEPTH, DIGIT_WIDTH> *cuda_stacks;

    cudaError_t error;

    /* Prime number allocation */
    error = cudaMalloc(&cuda_primes, bignum_sz * PRIMES_NUM); 
    if (error != cudaSuccess) {
        std::cerr << "Unable to allocate CUDA memory for prime storage" << std::endl;
        abort();
    }
    /* Prime trial allocation */
    error = cudaMalloc(&cuda_trials, bignum_sz * THREADS);
    if (error != cudaSuccess) {
        std::cerr << "Unable to allocate CUDA memory for prime trials" << std::endl;
        cudaFree(cuda_primes); abort();
    }
    /* Thread stack allocations */
    error = cudaMalloc(&cuda_stacks, bigstack_sz * THREADS);
    if (error != cudaSuccess) {
        std::cerr << "Unable to allocate CUDA memory for thread stacks" << std::endl;
        cudaFree(cuda_primes); cudaFree(cuda_trials); abort();
    }

    error = cudaMemcpy(cuda_primes, local_primes.data(),
                       bignum_sz * PRIMES_NUM, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Unable to copy prime numbers to CUDA memory" << std::endl;
        cudaFree(cuda_primes); cudaFree(cuda_trials); cudaFree(cuda_stacks); abort();
    }

    error = cudaMemcpy(cuda_trials, mr_trials.data(),
                       bignum_sz * THREADS, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Unable to copy trials numbers to CUDA memory" << std::endl;
        cudaFree(cuda_primes); cudaFree(cuda_trials); cudaFree(cuda_stacks); abort();
    }

    /*****************
    * RUN THE KERNEL *
    *****************/

    std::cout << "Running kernel" << std::endl;
    cudaDeviceSynchronize();
    MR_CUDA<STACK_DEPTH, DIGIT_WIDTH><<<GRID_SIZE, BLOCK_SIZE>>>(cuda_primes, cuda_trials, cuda_stacks);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel execution error..." << std::endl;
        cudaFree(cuda_primes); cudaFree(cuda_trials); cudaFree(cuda_stacks); abort();
    }

    /***********************
    * COPY AND FREE MEMORY *
    ***********************/

    cudaMemcpy(local_primes.data(), cuda_primes,
               bignum_sz * PRIMES_NUM, cudaMemcpyDeviceToHost);

    cudaFree(cuda_primes);
    cudaFree(cuda_trials);
    cudaFree(cuda_stacks);

    /*****************
    * OUTPUT RESULTS *
    *****************/

    cmp::bigint<DIGIT_WIDTH> *zero = local_stack.push();;
    int primecount = 0;
    cmp::to_int(0, zero);

    for (auto &p : local_primes) {
        if (cmp::compare(zero, &p) != 0) {
            cmp::print(&p);
            primecount++;
        }
    }

    local_stack.pop();

    std::cout << "Found " << primecount << " primes out of " << PRIMES_NUM 
              << " numbers tested." << std::endl;

    return true;
}

/*******************************************************************************
*                      SLOW PRIME (TRIAL DIVISION)                             *
*******************************************************************************/
/*******************************************************************************
*                      DO NOT USE. ONLY FOR DEVELOPMENT                        *
*******************************************************************************/

template<size_t D, size_t N>
__global__ void cudaPrimeFinder(cmp::bigint<N> *ps, bool *bs, cmp::env<D, N> *stacks)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x + gridDim.x;

    stacks[tid].sp = 0;
    bs[tid] = prime(&ps[tid], &stacks[tid]);

    sync();
}

void findPrimes(void)
{
    cudaError_t error;

    for (auto &p : local_primes) {
        cmp::rand_digits(&p, RAND_DIGITS);
    }
    
    cmp::bigint<DIGIT_WIDTH> *ps;
    cmp::env<STACK_DEPTH, DIGIT_WIDTH> *stacks;
    bool *bs;

    std::array<bool, THREADS> primeTrue;

    const size_t bignum_sz = sizeof(cmp::bigint<DIGIT_WIDTH>);
    const size_t bigstack_sz = sizeof(cmp::env<STACK_DEPTH, DIGIT_WIDTH>);

    cudaMalloc((void**)&ps, bignum_sz*local_primes.max_size());
    cudaMalloc((void**)&bs, sizeof(bool)*primeTrue.max_size());
    cudaMalloc((void**)&stacks, bigstack_sz*local_primes.max_size());

    cudaMemcpy(ps, local_primes.data(), bignum_sz*local_primes.max_size(), cudaMemcpyHostToDevice);
    
    cudaPrimeFinder<<<10, 128>>>(ps, bs, stacks);
   
    cudaMemcpy(primeTrue.data(), bs, sizeof(bool)*primeTrue.max_size(), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(error));

    cudaFree(ps);
    cudaFree(bs);
    cudaFree(stacks);
}
