/******************************************************************************
* File:             bignum_types.hpp
*
* Author:           Andy Belle-Isle  
* Created:          12/02/21 
* Description:      Multiprecision type and namespace declarations
*
* Email:            atb1317@rit.edu
*****************************************************************************/

#pragma once

#include <stdint.h>

namespace cmp {

/** @brief Big Number structure */
template<size_t N>
struct bigint {
    uint8_t digits[N]; /** Digit storage */
	int signbit;       /** 1 is positive, -1 if negative */
    int lastdigit;     /** Highest order digit index */
};

/** @brief Bignum Environment.
 * Holds a stack to store data
 */
template<size_t D, size_t N>
class env {
    public:
    bigint<N> data[D];  /* The "stack" to store temp values on */
    unsigned sp = 0;    /* Stack pointer */

    /** @brief "Pushes" a value onto the stack and returns the address.
     * If the stack is full, aborts.
     *
     * @return[bignum*] - Pointer to the top of the stack
     */
    bigint<N>* push(void)
    {
        if (sp >= D) {
            abort();
        }

        return &data[sp++];
    }

    /** @brief "Resets" the stack by moving the counter down 'p' indexes.
     * 
     * @param[p] - How many stack entries to pop
     */
    void pop(unsigned p = 1)
    {
        while (p-- > 0 && sp > 0)
            sp--;
    }
};

#define PLUS     1 // Positive sign value
#define MINUS   -1 // Negative sign value

} // namespace cmp
