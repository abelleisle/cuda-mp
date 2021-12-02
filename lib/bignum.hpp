#pragma once

#include <stdio.h>
#include <iostream>

#include "compile.hpp"

#define	MAXDIGITS	617    /* maximum length bignum */ 
//#define	MAXDIGITS	1235    /* maximum length bignum */ 
#define STACKDEPTH  12     /* How deep our bignum stack is */

#define PLUS		1       /* positive sign bit */
#define MINUS		-1      /* negative sign bit */

struct bignum {
    char digits[MAXDIGITS]; /* represent the number */
	int signbit;			/* 1 if positive, -1 if negative */ 
    int lastdigit;			/* index of high-order digit */
};

struct bignum_stack {
    bignum data[STACKDEPTH];
    int sp = 0;
};

FNC_DH void stack_init_bignum(bignum_stack *s)
{
    s->sp = 0;
}

FNC_H void print_bignum(bignum *n)
{
	int i;

	if (n->signbit == MINUS) printf("- ");
	for (i=n->lastdigit; i>=0; i--)
		printf("%c",'0'+ n->digits[i]);

	printf("\n");
}

FNC_DH void int_to_bignum(int s, bignum *n)
{
	int i;				/* counter */
	int t;				/* int to work with */

	if (s >= 0) n->signbit = PLUS;
	else n->signbit = MINUS;

	for (i=0; i<MAXDIGITS; i++) n->digits[i] = (char) 0;

	n->lastdigit = -1;

	t = abs(s);

	while (t > 0 && n->lastdigit < MAXDIGITS) {
		n->lastdigit ++;
		n->digits[ n->lastdigit ] = (t % 10);
		t = t / 10;
	}

	if (s == 0) n->lastdigit = 0;
}

FNC_DH void initialize_bignum(bignum *n)
{
	int_to_bignum(0,n);
}

/** @brief b = a */
FNC_DH void copy_bignum(bignum *a, bignum *b)
{
    b->lastdigit = a->lastdigit;
    b->signbit = a->signbit;
    *b = *a;
    //memcpy(&(b->digits), &(a->digits), MAXDIGITS*sizeof(char));
}

/* int max(int a, int b) */
/* { */
/* 	if (a > b) return(a); else return(b); */
/* } */

FNC_DH void zero_justify(bignum *n)
{
    if (n->lastdigit >= MAXDIGITS) n->lastdigit = MAXDIGITS - 1;

	while ((n->lastdigit > 0) && (n->digits[ n->lastdigit ] == 0))
		n->lastdigit --;

    if ((n->lastdigit == 0) && (n->digits[0] == 0))
		n->signbit = PLUS;	/* hack to avoid -0 */
}

FNC_DH void subtract_bignum(bignum *a, bignum *b, bignum *c);
FNC_DH int compare_bignum(bignum *a, bignum *b);

/*	c = a +- b;	*/
FNC_DH void add_bignum(bignum *a, bignum *b, bignum *c)
{
	int carry;			/* carry digit */
	int i;				/* counter */

	initialize_bignum(c);

	if (a->signbit == b->signbit) c->signbit = a->signbit;
	else {
		if (a->signbit == MINUS) {
			a->signbit = PLUS;
			subtract_bignum(b,a,c);
			a->signbit = MINUS;
		} else {
            b->signbit = PLUS;
            subtract_bignum(a,b,c);
            b->signbit = MINUS;
		}
		return;
	}

	c->lastdigit = max(a->lastdigit,b->lastdigit)+1;
    if (c->lastdigit >= MAXDIGITS) c->lastdigit = MAXDIGITS - 1;

	carry = 0;

	for (i=0; i<=(c->lastdigit); i++) {
		c->digits[i] = (char) (carry+a->digits[i]+b->digits[i]) % 10;
		carry = (carry + a->digits[i] + b->digits[i]) / 10;
	}

	zero_justify(c);
}

FNC_DH void add_i(bignum *a, int u, bignum_stack* stack)
{
    //int carry = u;
    bignum *tmp = &stack->data[stack->sp++];
    bignum *b   = &stack->data[stack->sp++];

    int_to_bignum(u, b);
    copy_bignum(a, tmp);

    add_bignum(tmp, b, a);

    stack->sp -= 2;
    
/*     a->lastdigit = 0; */
/*  */
/*     for (int i = 0; i < MAXDIGITS && carry != 0; i++) { */
/*         auto res = carry + a->digits[i]; */
/*         a->digits[i] = (char)(res%10); */
/*         carry = (char)(res/10); */
/*     } */

    zero_justify(a);
}

FNC_DH void subtract_bignum(bignum *a, bignum *b, bignum *c)
{
	int borrow;			/* has anything been borrowed? */
	int v;				/* placeholder digit */
	int i;				/* counter */

	initialize_bignum(c);

	if ((a->signbit == MINUS) || (b->signbit == MINUS)) {
        b->signbit = -1 * b->signbit;
        add_bignum(a,b,c);
        b->signbit = -1 * b->signbit;
		return;
    }

	if (compare_bignum(a,b) == PLUS) {
		subtract_bignum(b,a,c);
		c->signbit = MINUS;
		return;
	}

    c->lastdigit = max(a->lastdigit,b->lastdigit);
    if (c->lastdigit >= MAXDIGITS)
        c->lastdigit = MAXDIGITS - 1;
    borrow = 0;

    for (i=0; i<=(c->lastdigit); i++) {
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

FNC_DH int compare_bignum(bignum *a, bignum *b)
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


FNC_DH void digit_shift(bignum *n, int d)		/* multiply n by 10^d */
{
	int i;				/* counter */

	if ((n->lastdigit == 0) && (n->digits[0] == 0)) return;

	for (i=n->lastdigit; i>=0; i--)
		n->digits[i+d] = n->digits[i];

	for (i=0; i<d; i++) n->digits[i] = 0;

	n->lastdigit = n->lastdigit + d;
}



FNC_DH void multiply_bignum(bignum *a, bignum *b, bignum *c, bignum_stack *s)
{
	bignum *row = &s->data[s->sp++];			/* represent shifted row */
	bignum *tmp = &s->data[s->sp++];			/* placeholder bignum */
	int i,j;			/* counters */

	initialize_bignum(c);

	*row = *a;

	for (i=0; i<=b->lastdigit; i++) {
		for (j=1; j<=b->digits[i]; j++) {
			add_bignum(c,row,tmp);
			*c = *tmp;
		}
		digit_shift(row,1);
	}

	c->signbit = a->signbit * b->signbit;

	zero_justify(c);

    s->sp -= 2;
}


FNC_DH void divide_bignum(bignum *a, bignum *b, bignum *c, bignum *r, bignum_stack *s)
{
    bignum *row = &s->data[s->sp++];                     /* represent shifted row */
    bignum *tmp = &s->data[s->sp++];                     /* placeholder bignum */
	int asign, bsign;		/* temporary signs */
    int i;                        /* counters */

	initialize_bignum(c);

	c->signbit = a->signbit * b->signbit;

	asign = a->signbit;
	bsign = b->signbit;

	a->signbit = PLUS;
    b->signbit = PLUS;

	initialize_bignum(row);
	initialize_bignum(tmp);

	c->lastdigit = a->lastdigit;

    if (compare_bignum(row, b) == 0) {
    // If b == 0
        int_to_bignum(0, c); // Return 0
    } else if (compare_bignum(a, b) <= 0) {
    // a > 0
        for (i=a->lastdigit; i>=0; i--) {
            digit_shift(row,1);
            row->digits[0] = a->digits[i];
            c->digits[i] = 0;
            while (compare_bignum(row,b) != PLUS) {
                c->digits[i] ++;
                subtract_bignum(row,b,tmp);
                //row = tmp;
                copy_bignum(tmp, row);
            }
        }

        multiply_bignum(c, b, tmp, s);
        subtract_bignum(a, tmp, r);

        zero_justify(r);
        zero_justify(c);

        a->signbit = asign;
        b->signbit = bsign;
    } else {
    // if a < b
        initialize_bignum(c);
        copy_bignum(a, r);
    }

    s->sp -= 2;
}

/** @brief r = a % b */
FNC_DH void mod_bignum(bignum *a, bignum *b, bignum *r, bignum_stack *s)
{
    bignum *c = &s->data[s->sp++];

    divide_bignum(a,b,c,r,s);

    s->sp--;
}

FNC_DH bool even_bignum(bignum *a)
{
    return !(a->digits[0] & 1);
}

FNC_H void rand_bignum(bignum *a)
{
    for (int i = 0; i < MAXDIGITS; i++) {
        a->digits[i] = rand() % 10;
    }

    a->lastdigit = MAXDIGITS - 1;

    zero_justify(a);
}

FNC_H void rand_digits_bignum(bignum *a, unsigned digits)
{
    if (digits >= MAXDIGITS) digits = MAXDIGITS - 1;

    initialize_bignum(a);

    a->lastdigit = digits - 1;

    while (digits > 0) {
        a->digits[--digits] = rand() % 10;
    }

    zero_justify(a);
}

FNC_DH void rightshift_bignum(bignum *a, unsigned n)
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

/** @brief result = base^exp
 */
FNC_DH void pow_bignum(bignum *result, bignum *base, bignum *exp, bignum_stack *s)
{
    bignum *b = &s->data[s->sp++];
    bignum *i = &s->data[s->sp++];

    *b = *base;
    int_to_bignum(1, i);

    // If exp < 1
    if (compare_bignum(exp, i) > 0) {
        int_to_bignum(1, result);

    } else {
        *result = *base;

        // While i <= base
        while (compare_bignum(i, exp) > 0) {
            multiply_bignum(b, base, result, s);
            *b = *result;
            add_i(i, 1, s);
        }
    }

    zero_justify(result);

    s->sp -= 2;
}

/** @brief result = base^exp % mod
 */
FNC_DH void powmod_bignum(bignum *result, bignum *base, bignum *exp, bignum *mod, bignum_stack *s)
{
    /*
    bignum *r = &s->data[s->sp++];

    pow_bignum(result, base, exp, s);

    *r = *result;
    mod_bignum(r, mod, result, s);

    zero_justify(result);

    s->sp -= 1;
    */

    bignum *i = &s->data[s->sp++];
    bignum *zero = &s->data[s->sp++];

    initialize_bignum(zero); // zero = 0

    int_to_bignum(1, result);
    while (compare_bignum(exp,zero) != 0) {
        if (!even_bignum(exp)) {
            multiply_bignum(base, result, i, s);
            mod_bignum(i, mod, result, s);
        }

        multiply_bignum(base, base, i, s);
        mod_bignum(i, mod, base, s);

        rightshift_bignum(exp, 1);
    }

    zero_justify(result);

    s->sp -= 2;
}
