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

#include "bignum.hpp"
#include "bignum_prime.hpp"

#include <cstring>

#include <array>

int main(void)
{
    srand(time(NULL));

    bignum_stack stack;

    bignum p;
    rand_digits_bignum(&p, 15);

    print_bignum(&p);
    if (mr_bignum(&p, 32, &stack)) {
        std::cout << "Number is prime" << std::endl;
    } else {
        std::cout << "Not prime" << std::endl;
    }


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
