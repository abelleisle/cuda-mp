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

#include "bignum.hpp"
#include "bignum_prime.hpp"

const unsigned RAND_DIGITS = 30;
const unsigned PRIMES_NUM = 12800;
const unsigned BLOCK_SIZE = 32;
const unsigned MR_TRIALS  = BLOCK_SIZE;
const unsigned GRID_SIZE  = PRIMES_NUM;
const unsigned THREADS    = BLOCK_SIZE * GRID_SIZE;

std::array<bignum, GRID_SIZE> local_primes;

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

    /*********************
    * MEMORY USAGE STATS *
    *********************/

    size_t bignum_sz = sizeof(bignum);
    size_t bigstack_sz = sizeof(bignum_stack);

    std::cout << "Memory Usage statistics: " << std::endl;
    std::cout << " - MP Stack: " << sz(bigstack_sz) << std::endl;
    std::cout << " - Primes:   " << sz(bignum_sz * local_primes.max_size()) 
              << std::endl;

    /* Fill our random prime searches */
    std::cout << "Generating prime attempts..." << std::endl;
    for (auto &r : local_primes)
        rand_digits_bignum(&r, RAND_DIGITS);

    /********************
    * KERNEL OPERATIONS *
    ********************/

    bignum_stack stack;

    for (auto &r : local_primes) {
        if (mr_bignum(&r, 32, &stack)) {
            print_bignum(&r);
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

    /*

	int a,b,c;
	bignum n1,n2,n3,n4,zero;


    //rand_digits_bignum(&n1, 20);
    //print_bignum(&n1);

    //printf("Prime: %s\n", prime_bignum(&n1, &stack) ? "True" : "False");

	while (scanf("%d %d %d\n",&a,&b,&c) != EOF) {
		printf("a = %d    b = %d    c = %d\n",a,b,c);
		int_to_bignum(a,&n1);
		int_to_bignum(b,&n2);

		add_bignum(&n1,&n2,&n3);
		printf("addition -- ");
		print_bignum(&n3);

		printf("compare_bignum a ? b = %d\n",compare_bignum(&n1, &n2));

		subtract_bignum(&n1,&n2,&n3);
		printf("subtraction -- ");
		print_bignum(&n3);

        multiply_bignum(&n1,&n2,&n3,&stack);
		printf("multiplication -- ");
        print_bignum(&n3);

		int_to_bignum(0,&zero);
		if (compare_bignum(&zero, &n2) == 0)
			printf("division -- NaN \n");
        else {
			divide_bignum(&n1,&n2,&n3,&n4,&stack);
			printf("division -- ");
                    print_bignum(&n3);
            printf("remainder -- ");
                	print_bignum(&n4);
		}

        printf("prime -- %s\n", prime_bignum(&n1,&stack) ? "true" : "false");

        pow_bignum(&n3, &n1, &n2, &stack);
        printf("a^b -- ");
                print_bignum(&n3);

        int_to_bignum(c, &n3);
        powmod_bignum(&n4, &n1, &n2, &n3, &stack);
        printf("a^b mod c -- ");
                print_bignum(&n4);

        rightshift_bignum(&n1, b);
        printf("a >> b -- ");
                print_bignum(&n1);
		printf("--------------------------\n");
	}

    */

    return 0;
}
