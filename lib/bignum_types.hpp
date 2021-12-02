#pragma once

namespace bn {

/** @brief Big Number structure */
template<size_t N>
struct bn {
    char digits[N]; /* represent the number */
	int signbit;    /* 1 if positive, -1 if negative */ 
    int lastdigit;  /* index of high-order digit */
};

/** @brief Bignum Environment.
 * Holds a stack to store data
 */
template<size_t D, size_t N>
class env {
    bn<N> stack[D];  /* The "stack" to store temp values on */
    unsigned sp = 0; /* Stack pointer */

    /** @brief "Pushes" a value onto the stack and returns the address.
     * If the stack is full, aborts.
     *
     * @return[bignum*] - Pointer to the top of the stack
     */
    bn<N>* push(void)
    {
        if (sp >= D) {
            abort();
        }

        return &stack[sp++];
    }

    /** @brief "Resets" the stack by moving the counter down 'p' indexes.
     * 
     * @param[p] - How many stack entries to pop
     */
    void pop(unsigned p)
    {
        while (p-- > 0 && sp > 0)
            sp--;
    }
};

};
