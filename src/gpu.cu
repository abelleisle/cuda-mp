/******************************************************************************
* File:             gpu.cu
*
* Author:           Andy Belle-Isle  
* Created:          12/01/21 
* Description:      GPU (CUDA) host and device code for finding prime numbers
*                   using Miller-Rabin primality testing
*
* Email:            atb1317@rit.edu
*****************************************************************************/

#include "bignum.hpp"
#include "bignum_types.hpp"
#include "bignum_prime.hpp"

#include "prime_kernel.cu"

#include <chrono>
#include <array>
#include <functional>

int main()
{
    srand(time(NULL));
    //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);
    //findAdder();
    //findPrimes();
    //for (int i = 0; i < count; i++) {
    //    if (primeTrue[i])
    //        print_bignum(&primes[i]);
    //}

    using tp = std::chrono::high_resolution_clock;

    tp::time_point begin = tp::now();
    find_primes();
    tp::time_point end = tp::now();

    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
    double seconds = static_cast<double>(duration)/1000.0f;
    std::cout << "Elapsed Time: " << seconds << "s" << std::endl;

    return 0;

    /*
    int a,b;
    bignum n1,n2,n3,n4,zero;

    while (scanf("%d %d\n",&a,&b) != EOF) {
        printf("a = %d    b = %d\n",a,b);
        int_to_bignum(a,&n1);
        int_to_bignum(b,&n2);

        add_bignum(&n1,&n2,&n3);
        printf("addition -- ");
        print_bignum(&n3);

        printf("compare_bignum a ? b = %d\n",compare_bignum(&n1, &n2));

        subtract_bignum(&n1,&n2,&n3);
        printf("subtraction -- ");
        print_bignum(&n3);

                multiply_bignum(&n1,&n2,&n3);
        printf("multiplication -- ");
                print_bignum(&n3);

        int_to_bignum(0,&zero);
        if (compare_bignum(&zero, &n2) == 0)
            printf("division -- NaN \n");
                else {
            divide_bignum(&n1,&n2,&n3,&n4);
            printf("division -- ");
                    print_bignum(&n3);
            printf("remainder -- ");
                    print_bignum(&n4);
        }

        printf("prime -- %s\n", prime_bignum(&n1) ? "true" : "false");
        printf("--------------------------\n");
    }
    */
}
