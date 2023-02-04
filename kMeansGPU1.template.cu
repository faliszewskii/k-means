#if __INCLUDE_LEVEL__
#error "Don't include this file"
#endif

#include "kMeansGPU1.cuh"
#include "kMeansGPU1.cu"
#define D 3

template class KMeansGPU1Solver<D>;