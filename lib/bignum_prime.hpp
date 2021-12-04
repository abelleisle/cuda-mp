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
#include "bignum_types.hpp"
#include "compile.hpp"

namespace cmp {

template<size_t D, size_t N>
FNC_D bool prime(bigint<N> *p, env<D, N> *s)
{
    // Even or negative is false
    if (even(p) || p->signbit == MINUS)
        return false;

    bigint<N> *i = &s->data[s->sp++];
    bigint<N> *r = &s->data[s->sp++];
    bigint<N> *zero = &s->data[s->sp++];
    initialize(zero);

    to_int(1, i);
    if (compare(p, i) >= 0){
        return false;
    }

    to_int(3, i);
    if (compare(p, i) >= 0) {
        return true;
    }
    /* p % 3 == 0 */
    modulo(p, i, r, s);
    if (compare(r, zero) == 0)
        return false;

    sync();

    to_int(5, i);
    while(true) {
        multiply(i, i, r, s);
        if (compare(r, p) <= 0)
            break;

        sync();

        /* p % i == 0 */
        modulo(p, i, r, s);
        if (compare(r, zero) == 0)
            return false;

        sync();

        /* p % (i+2) == 0 */
        add_i(i, 2, s);
        modulo(p, i, r, s);
        if (compare(r, zero) == 0)
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

template<size_t D, size_t N>
FNC_D int mr_factor(bigint<N> *p, bigint<N> *d_ret, int *c_ret, env<D, N> *s)
{
    /* Even or negative - Not prime */
    if (even(p) || p->signbit == MINUS)
        return -1;

    /* If p <= 1 - Not prime */
    to_int(1, d_ret);
    if (compare(p, d_ret) >= 0)
        return -1;

    /* If p == 2 - Prime */
    to_int(2, d_ret);
    if (compare(p, d_ret) == 0)
        return 1;

    /* If p == 3 - Prime */
    to_int(3, d_ret);
    if (compare(p, d_ret) == 0)
        return 1;

    to_int(0, d_ret);
    
    // d = n - 1
    *d_ret = *p;
    add_i(d_ret, -1, s);

    sync();

    /* Factoring out 2 from d */
    *c_ret = 0;
    while(even(d_ret)) {
        (*c_ret)++;
        rightshift(d_ret, 1);
        sync();
    }

    return 0; // Regular return
}

template<size_t D, size_t N>
FNC_D void mr_treatrand(bigint<N> *p, bigint<N> *r, env<D, N> *s)
{
    bigint<N> *tmp   = &s->data[s->sp++];
    bigint<N> *tmp_p = &s->data[s->sp++];

    *tmp = *r;
    *tmp_p = *p;

    add_i(tmp_p, -4, s);
    sync();
    modulo(tmp, tmp_p, r, s);
    sync();
    add_i(tmp_p, 4, s);
    sync();
    add_i(r, 2, s);
    sync();

    s->sp -= 2;
}

template<size_t D, size_t N>
FNC_D int mr_innerloop(bigint<N> *p, bigint<N> *d, int c, bigint<N> *r, env<D, N> *s)
{
    bool maybe_prime = false;

    bigint<N> *result = &s->data[s->sp++];
    bigint<N> *tmp_i = &s->data[s->sp++];
    bigint<N> *two = &s->data[s->sp++];
    to_int(2, two);

    if (!maybe_prime) {
        // If remainder == 1
        to_int(1, tmp_i);
        powmod(result, r, d, p, s);
        if (compare(tmp_i,result) == 0) {
            maybe_prime = true;
        }
    }
    sync();

    if (!maybe_prime) {
        // If remainder == p-1
        *tmp_i = *p;
        add_i(tmp_i, -1, s);
        if (compare(tmp_i,result) == 0) {
            maybe_prime = true;
        }
    }
    sync();

    if (!maybe_prime) {
        for (int g = 1; g <= c - 1; g++) {
            *tmp_i = *r;
            powmod(result, tmp_i, two, p, s);
            to_int(1, tmp_i);
            if (compare(tmp_i, result) == 0) {
                maybe_prime = false; // This is not prime
                break;
            }

            *tmp_i = *p;
            add_i(tmp_i, -1, s);
            if (compare(tmp_i, result) == 0) {
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

template<size_t D, size_t N>
FNC_D bool mr(bigint<N> *p, int k, env<D, N> *s)
{
    /* Whether or not to continue searching. 
     * search = -1 - Number is not a prime
     * search =  0 - No prime found yet, keep searching
     * search =  1 - Number is a prime
     */
    int maybe_prime = 0;

    int c = 0;
    bigint<N> *d = &s->data[s->sp++]; // factor of 2
    bigint<N> *a = &s->data[s->sp++]; // random attempt

    maybe_prime = mr_factor(p, d, &c, s);

    if (maybe_prime == 0) {
        for (int j = 0; j < k; j++) {
            /* Generate random from 2 to p-2 */
            rand_digits(a, p->lastdigit+1);
            mr_treatrand(p, a, s);

            maybe_prime = mr_innerloop(p,d,c,a,s);
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

} // namespace cmp
