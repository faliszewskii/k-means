#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kMeansGPU2.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <math.h>

#define CUDART_INF_F 0x7ff0000000000000

#define CDR_ERR "cudaDeviceReset failed!"
#define CSD_ERR "cudaDeviceSet failed!"
#define CGLE_ERR "Kernel launch failed!"
#define CDS_ERR "cudaDeviceSynchronize returned error code"

template <int numberOfDimensions> void KMeansGPU2Solver<numberOfDimensions>::cudaCheck(cudaError_t status, char* message) {
	if (status == cudaSuccess)
		return;
	fprintf(stderr, "%s", message);
	exit(1);
}

struct is_equal_to
{
	is_equal_to(int number) {
		this->number = number;
	}

	__host__ __device__
		bool operator()(int x)
	{
		return x == number;
	}
	int number;
};

template <int numberOfDimensions> __device__ int findNearestClusterFor2(float* vector, float* centroidVectors, int centroidVectorLength)
{
	int minDistanceIndex = 0;
	float minDistanceSquared = CUDART_INF_F;
	float distanceSquared = 0;

	for (int i = 0; i < centroidVectorLength; i++) {
		distanceSquared = 0;
		for (int j = 0; j < numberOfDimensions; j++)
			distanceSquared += powf(vector[j] - centroidVectors[i * numberOfDimensions + j], 2);
		if (distanceSquared < minDistanceSquared) {
			minDistanceSquared = distanceSquared;
			minDistanceIndex = i;
		}
	}

	return minDistanceIndex;
}

template <int numberOfDimensions> __global__ void findNearestClustersKernel2(float* dataVectors, int* centroidMemberships, int* membershipChangeVector, float* centroidVectors, int centroidVectorLength, int dataVectorsLength)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= dataVectorsLength)
		return;

	int index = findNearestClusterFor2<numberOfDimensions>(&(dataVectors[tid * numberOfDimensions]), centroidVectors, centroidVectorLength);
	if (centroidMemberships[tid * numberOfDimensions] != index * numberOfDimensions) {
		membershipChangeVector[tid] = 1;
		for (int i = 0; i < numberOfDimensions; i++)
			centroidMemberships[tid * numberOfDimensions + i] = index * numberOfDimensions + i;
	}	
}

template <int numberOfDimensions> void KMeansGPU2Solver<numberOfDimensions>::solve()
{
	float membershipChangeFraction;
	int iteration = 0;

	do {
		iteration++;
		clearVariables();
		findNearestClusters();
		averageNewClusters();
	} while (iteration < limit && getMembershipChangeFraction() > threshold);

}


template <int numberOfDimensions> void KMeansGPU2Solver<numberOfDimensions>::initSolver(float* dataVectors, int dataVectorLength, int centroidVectorLength, float threshold, int limit)
{
	cudaCheck(cudaSetDevice(0), CSD_ERR);

	cudaMallocManaged(&(this->dataVectors), dataVectorLength * numberOfDimensions * sizeof(float));
	thrust::copy(dataVectors, dataVectors + dataVectorLength * numberOfDimensions, this->dataVectors);
		

	this->dataVectorLength = dataVectorLength;
	this->centroidVectorLength = centroidVectorLength;
	this->threshold = threshold;
	this->limit = limit;

	copiedData = new float[dataVectorLength * numberOfDimensions];
	copiedMemberships = new float[dataVectorLength * numberOfDimensions];
	centroids = new float[centroidVectorLength * numberOfDimensions];
	keys = new int[centroidVectorLength * numberOfDimensions];

	cudaMallocManaged(&membershipChangeVector, dataVectorLength * sizeof(int));
	cudaMallocManaged(&centroidMemberships, dataVectorLength * numberOfDimensions * sizeof(int));
	thrust::fill(centroidMemberships, centroidMemberships + dataVectorLength * numberOfDimensions, -1);

	cudaMallocManaged(&centroidVectors, centroidVectorLength * numberOfDimensions  * sizeof(float));

	cudaMallocManaged(&centroidKeys, centroidVectorLength * sizeof(int));

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = (this->dataVectors)[i * numberOfDimensions + j];

}

template <int numberOfDimensions> void KMeansGPU2Solver<numberOfDimensions>::clearVariables()
{
	thrust::fill(membershipChangeVector, membershipChangeVector + dataVectorLength, 0);
}

template <int numberOfDimensions> float KMeansGPU2Solver<numberOfDimensions>::getMembershipChangeFraction()
{
	int membershipChangeCounter = thrust::reduce(membershipChangeVector, membershipChangeVector + dataVectorLength);
	return (float)membershipChangeCounter / dataVectorLength;
}

template <int numberOfDimensions> void KMeansGPU2Solver<numberOfDimensions>::findNearestClusters()
{
	int blockSize = 1024;
	int blocks = dataVectorLength / 1024 + 1;
	findNearestClustersKernel2<numberOfDimensions><<<blocks, blockSize>>>(dataVectors, centroidMemberships, membershipChangeVector, centroidVectors, centroidVectorLength, dataVectorLength);
	cudaCheck(cudaGetLastError(), CGLE_ERR);
	cudaCheck(cudaDeviceSynchronize(), CDS_ERR);
}

template <int numberOfDimensions> void KMeansGPU2Solver<numberOfDimensions>::averageNewClusters()
{
	thrust::copy(dataVectors, dataVectors + dataVectorLength * numberOfDimensions, copiedData);
	thrust::copy(centroidMemberships, centroidMemberships + dataVectorLength * numberOfDimensions, copiedMemberships);
	thrust::sort_by_key(copiedMemberships, copiedMemberships + dataVectorLength * numberOfDimensions, copiedData);

	thrust::reduce_by_key(
		copiedMemberships,
		copiedMemberships + dataVectorLength * numberOfDimensions,
		copiedData,
		keys,
		centroids
	);

	for (int i = 0; i < centroidVectorLength; i++) {
		int centroidMembershipCount = thrust::count_if(copiedMemberships, copiedMemberships + dataVectorLength * numberOfDimensions, is_equal_to(i*numberOfDimensions));
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = centroids[i * numberOfDimensions + j] / centroidMembershipCount;
	}

}

template <int numberOfDimensions> void KMeansGPU2Solver<numberOfDimensions>::clearSolver()
{
	cudaCheck(cudaDeviceReset(), CDR_ERR);
}
