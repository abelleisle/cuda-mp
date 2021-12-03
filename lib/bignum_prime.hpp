/******************************************************************************
* File:             bignum_prime.hpp
*
* Author:           Andy Belle-Isle  
* Created:          12/01/21 
* Description:      Prime finding implementations for multi-precision numbers
*
* Email:            atb1317@rit.edu
*****************************************************************************/

#pragma once

#include "bignum.hpp"
#include "compile.hpp"

FNC_D bool prime_bignum(bignum *p, bignum_stack *s)
{
    // Even or negative is false
    if (even_bignum(p) || p->signbit == MINUS)
        return false;

    bignum *i = &s->data[s->sp++];
    bignum *r = &s->data[s->sp++];
    bignum *zero = &s->data[s->sp++];
    initialize_bignum(zero);

    int_to_bignum(1, i);
    if (compare_bignum(p, i) >= 0){
        return false;
    }

    int_to_bignum(3, i);
    if (compare_bignum(p, i) >= 0) {
        return true;
    }
    /* p % 3 == 0 */
    mod_bignum(p, i, r, s);
    if (compare_bignum(r, zero) == 0)
        return false;

    sync();

    int_to_bignum(5, i);
    while(true) {
        multiply_bignum(i, i, r, s);
        if (compare_bignum(r, p) <= 0)
            break;

        sync();

        /* p % i == 0 */
        mod_bignum(p, i, r, s);
        if (compare_bignum(r, zero) == 0)
            return false;

        sync();

        /* p % (i+2) == 0 */
        add_i(i, 2, s);
        mod_bignum(p, i, r, s);
        if (compare_bignum(r, zero) == 0)
            return false;

        sync();

        add_i(i, 4, s); /* i += 6 */
    }

    s->sp -= 3;

    return true;
}

/*******************************************************************************
*                                MILLER-RABIN                                  *
*******************************************************************************/

FNC_D int mr_bignum_factor(bignum *p, bignum *d_ret, int *c_ret, bignum_stack *s)
{
    /* Even or negative - Not prime */
    if (even_bignum(p) || p->signbit == MINUS)
        return -1;

    /* If p <= 1 - Not prime */
    int_to_bignum(1, d_ret);
    if (compare_bignum(p, d_ret) >= 0)
        return -1;

    /* If p == 2 - Prime */
    int_to_bignum(2, d_ret);
    if (compare_bignum(p, d_ret) == 0)
        return 1;

    /* If p == 3 - Prime */
    int_to_bignum(3, d_ret);
    if (compare_bignum(p, d_ret) == 0)
        return 1;

    int_to_bignum(0, d_ret);
    
    // d = n - 1
    *d_ret = *p;
    add_i(d_ret, -1, s);

    sync();

    /* Factoring out 2 from d */
    *c_ret = 0;
    while(even_bignum(d_ret)) {
        (*c_ret)++;
        rightshift_bignum(d_ret, 1);
        sync();
    }

    return 0; // Regular return
}

FNC_D void mr_bignum_treatrand(bignum *p, bignum *r, bignum_stack *s)
{
    bignum *tmp   = &s->data[s->sp++];
    bignum *tmp_p = &s->data[s->sp++];

    *tmp = *r;
    *tmp_p = *p;

    add_i(tmp_p, -4, s);
    sync();
    mod_bignum(tmp, tmp_p, r, s);
    sync();
    add_i(tmp_p, 4, s);
    sync();
    add_i(r, 2, s);
    sync();

    s->sp -= 2;
}

FNC_D int mr_bignum_innerloop(bignum *p, bignum *d, int c, bignum *r, bignum_stack *s)
{
    bool maybe_prime = false;

    bignum *result = &s->data[s->sp++];
    bignum *tmp_i = &s->data[s->sp++];
    bignum *two = &s->data[s->sp++];
    int_to_bignum(2, two);

    if (!maybe_prime) {
        // If remainder == 1
        int_to_bignum(1, tmp_i);
        powmod_bignum(result, r, d, p, s);
        if (compare_bignum(tmp_i,result) == 0) {
            maybe_prime = true;
        }
    }
    sync();

    if (!maybe_prime) {
        // If remainder == p-1
        *tmp_i = *p;
        add_i(tmp_i, -1, s);
        if (compare_bignum(tmp_i,result) == 0) {
            maybe_prime = true;
        }
    }
    sync();

    if (!maybe_prime) {
        for (int g = 1; g <= c - 1; g++) {
            *tmp_i = *r;
            powmod_bignum(result, tmp_i, two, p, s);
            int_to_bignum(1, tmp_i);
            if (compare_bignum(tmp_i, result) == 0) {
                maybe_prime = false; // This is not prime
                break;
            }

            *tmp_i = *p;
            add_i(tmp_i, -1, s);
            if (compare_bignum(tmp_i, result) == 0) {
                maybe_prime = true;
                break;
            }
        }
    }
    sync();

    s->sp -= 3;

    if (!maybe_prime)
        return -1; // Not prime

    return 1; // Prime
}

FNC_D bool mr_bignum(bignum *p, int k, bignum_stack *s)
{
    /* Whether or not to continue searching. 
     * search = -1 - Number is not a prime
     * search =  0 - No prime found yet, keep searching
     * search =  1 - Number is a prime
     */
    int maybe_prime = 0;

    int c = 0;
    bignum *d = &s->data[s->sp++]; // factor of 2
    bignum *a = &s->data[s->sp++]; // random attempt

    maybe_prime = mr_bignum_factor(p, d, &c, s);

    if (maybe_prime == 0) {
        for (int j = 0; j < k; j++) {
            /* Generate random from 2 to p-2 */
            rand_digits_bignum(a, p->lastdigit+1);
            mr_bignum_treatrand(p, a, s);

            maybe_prime = mr_bignum_innerloop(p,d,c,a,s);
            if (maybe_prime == 1)
                continue;
            if (maybe_prime == -1)
                break;
        } 
    }

    sync();

    s->sp -= 2;

    if (maybe_prime >= 0)
        return true;
    else
        return false;
}
