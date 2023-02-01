#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kMeansGPU1.cuh"
#include "device_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDART_INF_F 0x7ff0000000000000

#define CDR_ERR "cudaDeviceReset failed!"
#define CSD_ERR "cudaDeviceSet failed!"
#define CGLE_ERR "Kernel launch failed!"
#define CDS_ERR "cudaDeviceSynchronize returned error code"

void cudaCheck(cudaError_t status, char* message) {
	if (status == cudaSuccess)
		return;
	fprintf(stderr, "%s", message);
	exit(1);
}

__device__ int findNearestClusterFor(float* vector, float* centroidVectors, int numberOfDimensions, int centroidVectorLength)
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

__global__ void findNearestClustersKernel(float* dataVectors, int numberOfDimensions, int* centroidMemberships, int* membershipChangeCounter, float* newCentroidVectors, int* centroidMembershipCounts, float* centroidVectors, int centroidVectorLength, int dataVectorsLength)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= dataVectorsLength)
		return;

	int index = findNearestClusterFor(&dataVectors[tid * numberOfDimensions], centroidVectors, numberOfDimensions, centroidVectorLength);
	if (centroidMemberships[tid] != index) {
		atomicAdd(membershipChangeCounter, 1);
		centroidMemberships[tid] = index;
	}
	for (int j = 0; j < numberOfDimensions; j++)
		atomicAdd(&newCentroidVectors[index * numberOfDimensions + j], dataVectors[tid * numberOfDimensions + j]);
	atomicAdd(&centroidMembershipCounts[index], 1);
}

void KMeansGPU1Solver::solve()
{
	float membershipChangeFraction;

	do {
		clearVariables();
		findNearestClusters();
		averageNewClusters();
		membershipChangeFraction = (float)*membershipChangeCounter / dataVectorLength;
	} while (membershipChangeFraction > threshold);

}


void KMeansGPU1Solver::initSolver(float* dataVectors, int dataVectorLength, int numberOfDimensions, int centroidVectorLength, float threshold)
{
	cudaCheck(cudaSetDevice(0), CSD_ERR);

	cudaMallocManaged(&(this->dataVectors), dataVectorLength * numberOfDimensions * sizeof(float));
	for (int i = 0; i < dataVectorLength * numberOfDimensions; i++)
		this->dataVectors[i] = dataVectors[i];

	this->dataVectorLength = dataVectorLength;
	this->numberOfDimensions = numberOfDimensions;
	this->centroidVectorLength = centroidVectorLength;
	this->threshold = threshold;

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

void KMeansGPU1Solver::clearVariables()
{
	*membershipChangeCounter = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			newCentroidVectors[i * numberOfDimensions + j] = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		centroidMembershipCounts[i] = 0;
}

void KMeansGPU1Solver::findNearestClusters()
{
	int blockSize = 1024;
	int blocks = dataVectorLength / 1024 + 1;
	findNearestClustersKernel<<<blocks, blockSize>>>(dataVectors, numberOfDimensions, centroidMemberships, membershipChangeCounter, newCentroidVectors, centroidMembershipCounts, centroidVectors, centroidVectorLength, dataVectorLength);
	cudaCheck(cudaGetLastError(), CGLE_ERR);
	cudaCheck(cudaDeviceSynchronize(), CDS_ERR);
}

void KMeansGPU1Solver::averageNewClusters()
{
	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = newCentroidVectors[i * numberOfDimensions + j] / centroidMembershipCounts[i];
}

int KMeansGPU1Solver::findNearestClusterFor(float* vector)
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

void KMeansGPU1Solver::clearSolver()
{
	cudaCheck(cudaDeviceReset(), CDR_ERR);
}
