#if __INCLUDE_LEVEL__
#error "Don't include this file"
#endif

#include "kMeansCPU.h"
#include "kMeansCPU.cpp"
#define D 3

template class KMeansCPUSolver<D>;