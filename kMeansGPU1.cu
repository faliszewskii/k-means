#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kMeansGPU1.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <math.h>

#define CUDART_INF_F 0x7ff0000000000000

#define CDR_ERR "cudaDeviceReset failed!"
#define CSD_ERR "cudaDeviceSet failed!"
#define CGLE_ERR "Kernel launch failed!"
#define CDS_ERR "cudaDeviceSynchronize returned error code"

template <int numberOfDimensions> void KMeansGPU1Solver<numberOfDimensions>::cudaCheck(cudaError_t status, char* message) {
	if (status == cudaSuccess)
		return;
	fprintf(stderr, "%s", message);
	exit(1);
}

template <int numberOfDimensions> __device__ int findNearestClusterFor(float* vector, float* centroidVectors, int centroidVectorLength)
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

template <int numberOfDimensions> __global__ void findNearestClustersKernel(float* dataVectors, int* centroidMemberships, int* membershipChangeCounter, float* newCentroidVectors, int* centroidMembershipCounts, float* centroidVectors, int centroidVectorLength, int dataVectorsLength)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= dataVectorsLength)
		return;

	int index = findNearestClusterFor<numberOfDimensions>(&dataVectors[tid * numberOfDimensions], centroidVectors, centroidVectorLength);
	if (centroidMemberships[tid] != index) {
		atomicAdd(membershipChangeCounter, 1);
		centroidMemberships[tid] = index;
	}
	for (int j = 0; j < numberOfDimensions; j++)
		atomicAdd(&newCentroidVectors[index * numberOfDimensions + j], dataVectors[tid * numberOfDimensions + j]);
	atomicAdd(&centroidMembershipCounts[index], 1);
}

template <int numberOfDimensions> void KMeansGPU1Solver<numberOfDimensions>::solve()
{
	float membershipChangeFraction;
	int iteration = 0;

	do {
		clearVariables();
		findNearestClusters();
		averageNewClusters();
		membershipChangeFraction = (float)*membershipChangeCounter / dataVectorLength;
	} while (iteration < limit && membershipChangeFraction > threshold);

}


template <int numberOfDimensions> void KMeansGPU1Solver<numberOfDimensions>::initSolver(float* dataVectors, int dataVectorLength, int centroidVectorLength, float threshold, int limit)
{
	cudaCheck(cudaSetDevice(0), CSD_ERR);

	cudaMallocManaged(&(this->dataVectors), dataVectorLength * numberOfDimensions * sizeof(float));
	for (int i = 0; i < dataVectorLength * numberOfDimensions; i++)
		this->dataVectors[i] = dataVectors[i];

	this->dataVectorLength = dataVectorLength;
	this->centroidVectorLength = centroidVectorLength;
	this->threshold = threshold;
	this->limit = limit;

	cudaMallocManaged(&membershipChangeCounter, sizeof(int));
	*(membershipChangeCounter) = 0;

	cudaMallocManaged(&(centroidMemberships), dataVectorLength  * sizeof(int));
	cudaMallocManaged(&(centroidMembershipCounts), centroidVectorLength * sizeof(int));
	
	cudaMallocManaged(&(centroidVectors), centroidVectorLength * numberOfDimensions * sizeof(float));
	cudaMallocManaged(&(newCentroidVectors), centroidVectorLength * numberOfDimensions * sizeof(float));

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = dataVectors[i * numberOfDimensions + j];

}

template <int numberOfDimensions> void KMeansGPU1Solver<numberOfDimensions>::clearVariables()
{
	*membershipChangeCounter = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			newCentroidVectors[i * numberOfDimensions + j] = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		centroidMembershipCounts[i] = 0;
}

template <int numberOfDimensions> void KMeansGPU1Solver<numberOfDimensions>::findNearestClusters()
{
	int blockSize = 1024;
	int blocks = dataVectorLength / 1024 + 1;
	findNearestClustersKernel<numberOfDimensions><<<blocks, blockSize>>>(dataVectors, centroidMemberships, membershipChangeCounter, newCentroidVectors, centroidMembershipCounts, centroidVectors, centroidVectorLength, dataVectorLength);
	cudaCheck(cudaGetLastError(), CGLE_ERR);
	cudaCheck(cudaDeviceSynchronize(), CDS_ERR);
}

template <int numberOfDimensions> void KMeansGPU1Solver<numberOfDimensions>::averageNewClusters()
{
	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = newCentroidVectors[i * numberOfDimensions + j] / centroidMembershipCounts[i];
}

template <int numberOfDimensions> void KMeansGPU1Solver<numberOfDimensions>::clearSolver()
{
	cudaCheck(cudaDeviceReset(), CDR_ERR);
}
