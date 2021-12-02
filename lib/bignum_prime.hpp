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

FNC_D bool mr_bignum(bignum *p, int k, bignum_stack *s)
{
    /* Even or negative - Not prime */
    if (even_bignum(p) || p->signbit == MINUS)
        return false;

    bignum *i = &s->data[s->sp++];

    /* If p <= 1 - Not prime */
    int_to_bignum(1, i);
    if (compare_bignum(p, i) >= 0)
        return false;

    /* If p == 2 - Prime */
    int_to_bignum(2, i);
    if (compare_bignum(p, i) == 0)
        return true;

    /* If p == 3 - Prime */
    int_to_bignum(3, i);
    if (compare_bignum(p, i) == 0)
        return true;

    int_to_bignum(0, i);
    
    // d = n - 1
    bignum *d = &s->data[s->sp++];
    *d = *p;
    add_i(d, -1, s);

    sync();

    /* Factoring out 2 from d */
    int c = 0;
    while(true) {
        if (!even_bignum(d))
            break;

        c++;
        rightshift_bignum(d, 1);
        sync();
    }

    bignum *a = &s->data[s->sp++];
    bignum *r = &s->data[s->sp++];
    bignum *two = &s->data[s->sp++];
    int_to_bignum(2, two);

    for (int j = 0; j < k; j++) {
        /* Generate random from 2 to p-2 */
        rand_digits_bignum(a, p->lastdigit+1);
        add_i(a, 2, s);
        add_i(p, -2, s);
        mod_bignum(a,p,r,s);
        add_i(p, 2, s);
        *a = *r;
        sync();

        // If remainder == 1
        int_to_bignum(1, i);
        powmod_bignum(r, a, d, p, s);
        if (compare_bignum(i,r) == 0) {
            continue;
        }
        sync();

        // If remainder == p-1
        *i = *p;
        add_i(i, -1, s);
        if (compare_bignum(i,r) == 0) {
            continue;
        }
        sync();

        for (int g = 1; g <= c - 1; g++) {
            *i = *r;
            powmod_bignum(r, i, two, p, s);
            int_to_bignum(1, i);
            if (compare_bignum(i, r) == 0)
                return false;

            *i = *p;
            add_i(i, -1, s);
            if (compare_bignum(i, r) == 0)
                goto NEXTTRY;
        }

        return false;
    NEXTTRY:
        continue;
    } 
    // TODO: put this back
    // sync();

    // TODO: fix this, return false's don't reset the stack
    s->sp -= 5;

    return true;
}
