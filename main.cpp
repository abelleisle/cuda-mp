#include "bignum.hpp"

#include <iostream>

int main(void)
{
    bignum one;
    bignum two;
    bignum three;

    int_to_bignum(234234, &one);
    int_to_bignum(-92831, &two);

    std::cout << "a = "; print_bignum(&one);
    std::cout << "b = "; print_bignum(&two);

    add_bignum(&one, &two, &three);

    std::cout << "--\n"
                 "a + b = "; print_bignum(&three);


    return 0;
}
