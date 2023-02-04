#if __INCLUDE_LEVEL__
#error "Don't include this file"
#endif

#include "kMeansGPU2.cuh"
#include "kMeansGPU2.cu"
#define D 3

template class KMeansGPU2Solver<D>;