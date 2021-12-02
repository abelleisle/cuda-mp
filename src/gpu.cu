/******************************************************************************
* File:             gpu.cu
*
* Author:           Andy Belle-Isle  
* Created:          12/01/21 
* Description:      GPU (CUDA) host and device code for finding prime numbers
*                   using Miller-Rabin primality testing
*
* Email:            atb1317@rit.edu
*****************************************************************************/

#include "bignum.hpp"
#include "bignum_prime.hpp"

#include <array>

const size_t count = 1280;

std::array<bignum, count> primes;
std::array<bool  , count> primeTrue;

__global__ void cudaAdder(bignum *a, bignum *b, bignum *c)
{
    if (threadIdx.x == 0)
        add_bignum(a, b, c);
    sync();
}

void findAdder(void)
{
    bignum a;
    bignum b;
    bignum x;

    std::cout << "Sizeof(bignum): " << sizeof(bignum) << std::endl;

    std::cout << "CUDA: " << std::endl;
    rand_digits_bignum(&a, 610);
    rand_digits_bignum(&b, 600);
    initialize_bignum(&x);

    print_bignum(&a);
    print_bignum(&b);

    bignum *c;
    bignum *d;
    bignum *z;

    cudaMalloc((void**)&c, sizeof(bignum));
    cudaMalloc((void**)&d, sizeof(bignum));
    cudaMalloc((void**)&z, sizeof(bignum));

    cudaMemcpy(c, &a, sizeof(bignum), cudaMemcpyHostToDevice);
    cudaMemcpy(d, &b, sizeof(bignum), cudaMemcpyHostToDevice);
    cudaMemcpy(z, &x, sizeof(bignum), cudaMemcpyHostToDevice);

    cudaAdder<<<1,1>>>(c, d, z);

    cudaMemcpy(&a, c, sizeof(bignum), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, d, sizeof(bignum), cudaMemcpyDeviceToHost);
    cudaMemcpy(&x, z, sizeof(bignum), cudaMemcpyDeviceToHost);

    cudaFree(c);
    cudaFree(d);
    cudaFree(z);

    print_bignum(&x);

    std::cout << "CPU: " << std::endl;
    print_bignum(&a);
    print_bignum(&b);
    add_bignum(&a, &b, &x);
    print_bignum(&x);
}

__global__ void cudaPrimeFinder(bignum *ps, bool *bs, bignum_stack *stacks)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x + gridDim.x;

    stack_init_bignum(&stacks[tid]);
    bs[tid] = prime_bignum(&ps[tid], &stacks[tid]);

    sync();
}

void findPrimes(void)
{
    cudaError_t error;

    for (auto &p : primes) {
        rand_digits_bignum(&p, 10);
    }
    
    bignum *ps;
    bignum_stack *stacks;
    bool *bs;

    cudaMalloc((void**)&ps, sizeof(bignum)*primes.max_size());
    cudaMalloc((void**)&bs, sizeof(bool)*primeTrue.max_size());
    cudaMalloc((void**)&stacks, sizeof(bignum_stack)*primes.max_size());

    cudaMemcpy(ps, primes.data(), sizeof(bignum)*primes.max_size(), cudaMemcpyHostToDevice);
    
    cudaPrimeFinder<<<10, 128>>>(ps, bs, stacks);
   
    cudaMemcpy(primeTrue.data(), bs, sizeof(bool)*primeTrue.max_size(), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(error));

    cudaFree(ps);
    cudaFree(bs);
    cudaFree(stacks);
}

int main()
{
    srand(time(NULL));
    //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);
    //findAdder();
    findPrimes();
    for (int i = 0; i < count; i++) {
        if (primeTrue[i])
            print_bignum(&primes[i]);
    }

    return 0;

    /*
	int a,b;
	bignum n1,n2,n3,n4,zero;

	while (scanf("%d %d\n",&a,&b) != EOF) {
		printf("a = %d    b = %d\n",a,b);
		int_to_bignum(a,&n1);
		int_to_bignum(b,&n2);

		add_bignum(&n1,&n2,&n3);
		printf("addition -- ");
		print_bignum(&n3);

		printf("compare_bignum a ? b = %d\n",compare_bignum(&n1, &n2));

		subtract_bignum(&n1,&n2,&n3);
		printf("subtraction -- ");
		print_bignum(&n3);

                multiply_bignum(&n1,&n2,&n3);
		printf("multiplication -- ");
                print_bignum(&n3);

		int_to_bignum(0,&zero);
		if (compare_bignum(&zero, &n2) == 0)
			printf("division -- NaN \n");
                else {
			divide_bignum(&n1,&n2,&n3,&n4);
			printf("division -- ");
                	print_bignum(&n3);
            printf("remainder -- ");
                	print_bignum(&n4);
		}

        printf("prime -- %s\n", prime_bignum(&n1) ? "true" : "false");
		printf("--------------------------\n");
	}
    */
}
