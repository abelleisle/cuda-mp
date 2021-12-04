/******************************************************************************
* File:             bignum.hpp
*
* Author:           Andy Belle-Isle  
* Created:          11/28/21 
* Description:      Multi-precision arithmetic library
*
* Email:            atb1317@rit.edu
*****************************************************************************/

#pragma once

#include <stdio.h>
#include <iostream>

#include "bignum_types.hpp"
#include "compile.hpp"

namespace cmp {

/** @brief prints the value of the bigint to terminal
 * @param[n] - The bigint to print
 */
template<size_t N>
FNC_H void print(bigint<N> *n)
{
	int i;

	if (n->signbit == MINUS) printf("- ");
	for (i=n->lastdigit; i>=0; i--)
		printf("%c",'0'+ n->digits[i]);

	printf("\n");
}

/** @brief Writes the value of an integer to the bigint
 * @param[s] - Integer to write
 * @param[n] - Bigint to modify
 */
template<size_t N>
FNC_DH void to_int(int s, bigint<N> *n)
{
	if (s >= 0) n->signbit = PLUS;
    else        n->signbit = MINUS;

	for (size_t i = 0; i < N; i++)
        n->digits[i] = 0;

	n->lastdigit = -1;

	int t = abs(s);

	while (t > 0 && n->lastdigit < (int)N) {
		n->lastdigit ++;
		n->digits[ n->lastdigit ] = (t % 10);
		t = t / 10;
	}

	if (s == 0)
        n->lastdigit = 0;
}

/** @brief Initializes the bignum.
 * @note For now this calls to_int on 'n'.
 * @param[n] - The bigint to initialize
 */
template<size_t N>
FNC_DH void initialize(bigint<N> *n)
{
	to_int(0, n);
}

/** @brief Copies one bigint to another 
 * @param[dst] - Destination
 * @param[src] - Source
 */
template<size_t N>
FNC_DH void copy(bigint<N> *dst, bigint<N> *src)
{
    dst->lastdigit = src->lastdigit;
    dst->signbit = src->signbit;
    *dst = *src;
    //memcpy(&(b->digits), &(a->digits), MAXDIGITS*sizeof(char));
}

/** @brief Justifies data within the bigint to set the sign and last digit
 * @param[n] - The bigint to justify
 */
template<size_t N>
FNC_DH void zero_justify(bigint<N> *n)
{
    if (n->lastdigit >= (int)N) n->lastdigit = N - 1;

	while ((n->lastdigit > 0) && (n->digits[ n->lastdigit ] == 0))
		n->lastdigit --;

    if ((n->lastdigit == 0) && (n->digits[0] == 0))
		n->signbit = PLUS;	/* hack to avoid -0 */
}

template<size_t N>
FNC_DH void subtract(bigint<N> *a, bigint<N> *b, bigint<N> *c);
template<size_t N>
FNC_DH int compare(bigint<N> *a, bigint<N> *b);

/** @brief Adds two bigints together. c = a + b;
 * @param[a] - Input a
 * @param[b] - Input b
 * @param[c] - Output c
 */
template<size_t N>
FNC_DH void add(bigint<N> *a, bigint<N> *b, bigint<N> *c)
{
	int carry = 0;

	initialize(c);

	if (a->signbit == b->signbit) c->signbit = a->signbit;
	else {
		if (a->signbit == MINUS) {
			a->signbit = PLUS;
			subtract(b,a,c);
			a->signbit = MINUS;
		} else {
            b->signbit = PLUS;
            subtract(a,b,c);
            b->signbit = MINUS;
		}
		return;
	}

	c->lastdigit = max(a->lastdigit,b->lastdigit)+1;

    if (c->lastdigit >= (int)N) c->lastdigit = N - 1;

	for (int i=0; i <= (c->lastdigit); i++) {
		c->digits[i] = (char) (carry+a->digits[i]+b->digits[i]) % 10;
		carry = (carry + a->digits[i] + b->digits[i]) / 10;
	}

	zero_justify(c);
}

/** @brief Adds an integer value to a bigint. a = a + i
 * @param[a] - Bigint input/output
 * @param[i] - Input integer to add/subtract
 * @param[stack] - The cmp environment stack
 */
template<size_t D, size_t N>
FNC_DH void add_i(bigint<N> *a, int i, env<D,N>* stack)
{
    //int carry = u;
    bigint<N> *tmp = &stack->data[stack->sp++];
    bigint<N> *b   = &stack->data[stack->sp++];

    to_int(i, b);
    copy(tmp, a);

    add(tmp, b, a);

    stack->sp -= 2;
    
    /*
    a->lastdigit = 0;

    for (int i = 0; i < MAXDIGITS && carry != 0; i++) {
        auto res = carry + a->digits[i];
        a->digits[i] = (char)(res%10);
        carry = (char)(res/10);
    }
    */

    zero_justify(a);
}

/** @brief Subtracts one bigint from another. c = a - b
 * @param[a] - Input a
 * @param[b] - Input b
 * @param[c] - Output c
 */
template<size_t N>
FNC_DH void subtract(bigint<N> *a, bigint<N> *b, bigint<N> *c)
{
	int borrow = 0; // Borrow/carry
	int v; // Placeholder

	initialize(c);

	if ((a->signbit == MINUS) || (b->signbit == MINUS)) {
        b->signbit = -1 * b->signbit;
        add(a,b,c);
        b->signbit = -1 * b->signbit;
		return;
    }

	if (compare(a,b) == PLUS) {
		subtract(b,a,c);
		c->signbit = MINUS;
		return;
	}

    c->lastdigit = max(a->lastdigit,b->lastdigit);
    if (c->lastdigit >= (int)N)
        c->lastdigit = N - 1;

    for (int i = 0; i <= (c->lastdigit); i++) {
        v = (a->digits[i] - borrow - b->digits[i]);
        if (a->digits[i] > 0)
            borrow = 0;
        if (v < 0) {
            v = v + 10;
            borrow = 1;
        }

        c->digits[i] = (char) v % 10;
    }

	zero_justify(c);
}

/** @brief Compares the value of two bigints. a cmp b
 * @param[a] - Input a
 * @param[b] - Input b
 * @return[int] - The relationship between a and b.
 *                < 0, means A > B. > 0 means A < B. 0 means A == B.
 *                These can be combined.
 */
template<size_t N>
FNC_DH int compare(bigint<N> *a, bigint<N> *b)
{
	int i;				/* counter */

	if ((a->signbit == MINUS) && (b->signbit == PLUS)) return(PLUS);
	if ((a->signbit == PLUS) && (b->signbit == MINUS)) return(MINUS);

	if (b->lastdigit > a->lastdigit) return (PLUS * a->signbit);
	if (a->lastdigit > b->lastdigit) return (MINUS * a->signbit);

	for (i = a->lastdigit; i>=0; i--) {
		if (a->digits[i] > b->digits[i]) return(MINUS * a->signbit);
		if (b->digits[i] > a->digits[i]) return(PLUS * a->signbit);
	}

	return(0);
}


/** @brief "Shifts" the bigint binary over by 'd' to the right
 * @param[n] - Input bigint
 * @param[d] - How many bits to shift it to the right
 */
template<size_t N>
FNC_DH void digit_shift(bigint<N> *n, int d)
{
	if ((n->lastdigit == 0) && (n->digits[0] == 0)) return;

	for (int i = n->lastdigit; i >= 0; i--)
        n->digits[i+d] = n->digits[i];

	for (int i = 0; i < d; i++) n->digits[i] = 0;

	n->lastdigit = n->lastdigit + d;
}


/** @brief Multiply two bigints together. c = a * b
 * @param[a] - Input a
 * @param[b] - Input b
 * @param[c] - Output c
 * @param[s] - cmp environment
 */
template<size_t D, size_t N>
FNC_DH void multiply(bigint<N> *a, bigint<N> *b, bigint<N> *c, env<D, N> *s)
{
	bigint<N> *row = &s->data[s->sp++];			/* represent shifted row */
	bigint<N> *tmp = &s->data[s->sp++];			/* placeholder bigint<N> */

	initialize(c);

	*row = *a;

	for (int i = 0; i <= b->lastdigit; i++) {
		for (int j = 1; j <= b->digits[i]; j++) {
			add(c,row,tmp);
			*c = *tmp;
		}
		digit_shift(row,1);
	}

	c->signbit = a->signbit * b->signbit;

	zero_justify(c);

    s->sp -= 2;
}


/** @brief Divides one bigint by another. c = a / b.
 * @param[a] - Numerator
 * @param[b] - Demoninator
 * @param[c] - Quotient
 * @param[r] - Remainder
 * @param[s] - cmp environment
 */ 
template<size_t D, size_t N>
FNC_DH void divide(bigint<N> *a, bigint<N> *b, bigint<N> *c, bigint<N> *r, env<D, N> *s)
{
    bigint<N> *row = &s->data[s->sp++];
    bigint<N> *tmp = &s->data[s->sp++];

	initialize(c);

	c->signbit = a->signbit * b->signbit;

	int asign = a->signbit;
	int bsign = b->signbit;

	a->signbit = PLUS;
    b->signbit = PLUS;

	initialize(row);
	initialize(tmp);

	c->lastdigit = a->lastdigit;

    if (compare(row, b) == 0) {
    // If b == 0
        to_int(0, c); // Return 0
    } else if (compare(a, b) <= 0) {
    // a > 0
        for (int i = a->lastdigit; i >= 0; i--) {
            digit_shift(row,1);
            row->digits[0] = a->digits[i];
            c->digits[i] = 0;
            while (compare(row,b) != PLUS) {
                c->digits[i] ++;
                subtract(row,b,tmp);
                //row = tmp;
                copy(row, tmp);
            }
        }

        multiply(c, b, tmp, s);
        subtract(a, tmp, r);

        zero_justify(r);
        zero_justify(c);

        a->signbit = asign;
        b->signbit = bsign;
    } else {
    // if a < b
        initialize(c);
        copy(r, a);
    }

    s->sp -= 2;
}

/** @brief Performs a modulus on two bigints. r = a % b.
 * @param[a] - Numerator
 * @param[b] - Demonintor
 * @param[r] - Remainder/Output
 * @param[s] - cmp environment
 */
template<size_t D, size_t N>
FNC_DH void modulo(bigint<N> *a, bigint<N> *b, bigint<N> *r, env<D, N> *s)
{
    bigint<N> *c = &s->data[s->sp++];

    divide(a,b,c,r,s);

    s->sp--;
}

/** @brief Determines if bigint is even or not
 * @param[a] - Bigint to test
 * @return[bool] - True if even, false if not
 */
template<size_t N>
FNC_DH bool even(bigint<N> *a)
{
    return (a->digits[0] & 1) == 0;
}

/** @brief Generates a random number from 0 to the max range of the bigint
 * @param[a] - Bigint to randomize
 */
template<size_t N>
FNC_H void rand(bigint<N> *a)
{
    for (size_t i = 0; i < N; i++) {
        a->digits[i] = ::rand() % 10;
    }

    a->lastdigit = N - 1;

    zero_justify(a);
}

/** @brief Generates a random number with the given number of digits
 * @param[a] - Bigint to randomize
 * @param[digits] - How many digits to randomize
 */
template<size_t N>
FNC_H void rand_digits(bigint<N> *a, unsigned digits)
{
    if (digits >= N)
        digits = N - 1;

    initialize(a);

    a->lastdigit = digits - 1;

    while (digits > 0) {
        a->digits[--digits] = ::rand() % 10;
    }

    zero_justify(a);
}

/** @brief "Shifts" the bigint right a single bit as if it was binary
 * @param[a] - Bigint to shift
 * @param[n] - How many bits to shift
 */
template<size_t N>
FNC_DH void rightshift(bigint<N> *a, unsigned n)
{
    unsigned tmp,carry = 0;
    while (n--) {
        for (int i = a->lastdigit; i >= 0; i--) {
            tmp = (a->digits[i] + carry) >> 1;
            carry = (a->digits[i] & 1) * 10;

            a->digits[i] = tmp % 10;
        }
        zero_justify(a);
    }
}

/** @brief Performs power operation on bigints result = base^exp
 * @param[result] - Result output
 * @param[base] - Exponent base
 * @param[exp] - Exponent
 * @param[s] - cmp environment
 */
template<size_t D, size_t N>
FNC_DH void pow(bigint<N> *result, bigint<N> *base, bigint<N> *exp, env<D, N> *s)
{
    bigint<N> *b = &s->data[s->sp++];
    bigint<N> *i = &s->data[s->sp++];

    *b = *base;
    to_int(1, i);

    // If exp < 1
    if (compare(exp, i) > 0) {
        to_int(1, result);

    } else {
        *result = *base;

        // While i <= base
        while (compare(i, exp) > 0) {
            multiply(b, base, result, s);
            *b = *result;
            add_i(i, 1, s);
        }
    }

    zero_justify(result);

    s->sp -= 2;
}

/** @brief Performs powmod on bignums. result = base^exp % mod
 * Uses right-to-left binary exponentiation for a code speedup.
 * @param[result] - Result
 * @param[base] - Exponent base
 * @param[exp] - Exponent
 * @param[mod] - Modulus
 * @param[s] - cmp stack
 */
template<size_t D, size_t N>
FNC_DH void powmod(bigint<N> *result, bigint<N> *base, bigint<N> *exp, bigint<N> *mod, env<D, N> *s)
{
    bigint<N> *i = &s->data[s->sp++];
    bigint<N> *zero = &s->data[s->sp++];

    initialize(zero); // zero = 0

    to_int(1, result);
    while (compare(exp,zero) != 0) {
        if (!even(exp)) {
            multiply(base, result, i, s);
            modulo(i, mod, result, s);
        }

        multiply(base, base, i, s);
        modulo(i, mod, base, s);

        rightshift(exp, 1);
    }

    zero_justify(result);

    s->sp -= 2;
}

} // namespace cmp
