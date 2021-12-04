/******************************************************************************
* File:             cpu.cpp
*
* Author:           Andy Belle-Isle
* Created:          11/30/21 
* Description:      CPU "device" code to find large prime numbers using both
*                   Miller-Rabin and standard prime finding techniques.
*
* Email:            atb1317@rit.edu
*****************************************************************************/

#include <cstring>
#include <array>
#include <chrono>
#include <cmath>
#include <string>

#include "bignum.hpp"
#include "bignum_types.hpp"
#include "bignum_prime.hpp"

#define DIGIT_WIDTH 1236
#define STACK_DEPTH 15

const unsigned RAND_DIGITS = 20;
const unsigned PRIMES_NUM = 25600;
const unsigned BLOCK_SIZE = 32;
const unsigned MR_TRIALS  = BLOCK_SIZE;
const unsigned GRID_SIZE  = PRIMES_NUM;
const unsigned THREADS    = BLOCK_SIZE * GRID_SIZE;

std::array<cmp::bigint<DIGIT_WIDTH>, GRID_SIZE> local_primes;

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

bool find_primes(void)
{
    srand(time(NULL));
    cmp::env<STACK_DEPTH, DIGIT_WIDTH> stack;

    /*********************
    * MEMORY USAGE STATS *
    *********************/

    size_t bignum_sz = sizeof(cmp::bigint<DIGIT_WIDTH>);
    size_t bigstack_sz = sizeof(cmp::env<STACK_DEPTH, DIGIT_WIDTH>);

    std::cout << "Memory Usage statistics: " << std::endl;
    std::cout << " - MP Stack: " << sz(bigstack_sz) << std::endl;
    std::cout << " - Primes:   " << sz(bignum_sz * local_primes.max_size()) 
              << std::endl;

    /* Fill our random prime searches */
    std::cout << "Generating prime attempts..." << std::endl;
    for (auto &r : local_primes) {
        cmp::rand_digits(&r, RAND_DIGITS);
        if (cmp::even(&r))
            cmp::add_i(&r, 1, &stack);
    }

    /********************
    * KERNEL OPERATIONS *
    ********************/

    std::cout << "Running kernel.." << std::endl;
    for (auto &r : local_primes) {
        if (cmp::mr(&r, 32, &stack)) {
            cmp::print(&r);
        }
    }

    return true;
}

int main(void)
{
    using tp = std::chrono::high_resolution_clock;

    tp::time_point begin = tp::now();
    find_primes();
    tp::time_point end = tp::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
    double seconds = static_cast<double>(duration)/1000.0f;
    std::cout << "Elapsed Time: " << seconds << "s" << std::endl;

    return 0;
}
