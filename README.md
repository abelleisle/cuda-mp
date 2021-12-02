# cuda-mp
CUDA Multi-Precision Library (Used for prime finding)

## How does it work?
This Multi-precision library works by defining all numbers as BCD types. Each digit is stored as a (char) index in an array, with the values from 0-9 (BCD).

### Advantages
1. Makes it much easier to program/understand

### Disadvantages
1. Since each digit is stored in Base10 instead of Base2^32 (for a uint32_t), there are a lot of wasted bits.
e.g.: 2048-bit arithmetic requires at least 1234 bytes, one for each digit. While 2048-bit arithmetic using bit storage (non-bcd) only requires 64 bytes.
While the number storage only requires half that (617 digits, and 32 bytes respectively), double width is needed for multiplication operations.

## CUDA Speedup
In order for this code to provide a speedup over standard CPU-based bigint libraries, it **must** be run in parallel across many CUDA threads.
Each operation runs serially on a single thread which means that *multiple* operations should be performed at once with different numbers across multiple threads.

## FAQ

* Why are the early commits so bunched up and funky?

  - I started working on this project without VCS, and used my [Seafile](https://github.com/haiwen/seafile) server to
sync the code across devices. The early commits are essentially just snapshots of the code directory over the last week,
just condensed into a few commits for sake of versioning with VCS.
