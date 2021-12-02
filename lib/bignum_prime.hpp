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
