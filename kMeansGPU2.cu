#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kMeansGPU2.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <math.h>

#define CUDART_INF_F 0x7ff0000000000000

#define CDR_ERR "cudaDeviceReset failed!"
#define CSD_ERR "cudaDeviceSet failed!"
#define CGLE_ERR "Kernel launch failed!"
#define CDS_ERR "cudaDeviceSynchronize returned error code"

void cudaCheck2(cudaError_t status, char* message) {
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

__device__ int findNearestClusterFor2(float* vector, float* centroidVectors, int numberOfDimensions, int centroidVectorLength)
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

__global__ void findNearestClustersKernel2(float* dataVectors, int numberOfDimensions, int* centroidMemberships, thrust::device_ptr<int> membershipChangeVector, float* newCentroidVectors, thrust::device_ptr<int> thrust_centroidMembershipCounts, float* centroidVectors, int centroidVectorLength, int dataVectorsLength)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= dataVectorsLength)
		return;

	int index = findNearestClusterFor2(&dataVectors[tid * numberOfDimensions], centroidVectors, numberOfDimensions, centroidVectorLength);
	if (centroidMemberships[tid] != index) {
		membershipChangeVector[tid] = 1;
		//atomicAdd(membershipChangeCounter, 1); // Ten atomicAdd mo¿na pozbyæ siê poprzez reduce jedynek, które tutaj przypiszemy.
		centroidMemberships[tid] = index;
	}
	for (int j = 0; j < numberOfDimensions; j++)
		// Ten atomicAdd 
		atomicAdd(&newCentroidVectors[index * numberOfDimensions + j], dataVectors[tid * numberOfDimensions + j]);	
	thrust_centroidMembershipCounts[tid] = index;
}

void KMeansGPU2Solver::solve()
{
	float membershipChangeFraction;

	do {
		clearVariables();
		findNearestClusters();
		averageNewClusters();
	} while (getMembershipChangeFraction() > threshold);

}


void KMeansGPU2Solver::initSolver(float* dataVectors, int dataVectorLength, int numberOfDimensions, int centroidVectorLength, float threshold)
{
	cudaCheck2(cudaSetDevice(0), CSD_ERR);

	cudaMallocManaged(&(this->dataVectors), dataVectorLength * numberOfDimensions * sizeof(float));
	for (int i = 0; i < dataVectorLength * numberOfDimensions; i++)
		this->dataVectors[i] = dataVectors[i];

	this->dataVectorLength = dataVectorLength;
	this->numberOfDimensions = numberOfDimensions;
	this->centroidVectorLength = centroidVectorLength;
	this->threshold = threshold;

	int* membershipChangeVector;
	cudaMallocManaged(&membershipChangeVector, dataVectorLength * sizeof(int));
	thrust_MembershipChangeVector = thrust::device_ptr<int>(membershipChangeVector);
	

	cudaMallocManaged(&(centroidMemberships), dataVectorLength * sizeof(int));
	int* centroidMembershipCounts;
	cudaMallocManaged(&(centroidMembershipCounts), dataVectorLength * sizeof(int));
	thrust_centroidMembershipCounts = thrust::device_ptr<int>(centroidMembershipCounts);

	cudaMallocManaged(&(centroidVectors), centroidVectorLength * numberOfDimensions * sizeof(float));
	cudaMallocManaged(&(newCentroidVectors), centroidVectorLength * numberOfDimensions * sizeof(float));

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = dataVectors[i * numberOfDimensions + j];

}

void KMeansGPU2Solver::clearVariables()
{
	thrust::fill(thrust_MembershipChangeVector, thrust_MembershipChangeVector + dataVectorLength, 0);
	thrust::fill(thrust_centroidMembershipCounts, thrust_centroidMembershipCounts + dataVectorLength, 0);
	

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			newCentroidVectors[i * numberOfDimensions + j] = 0;
}

float KMeansGPU2Solver::getMembershipChangeFraction() 
{
	int membershipChangeCounter = thrust::reduce(thrust_MembershipChangeVector, thrust_MembershipChangeVector + dataVectorLength);
	return (float)membershipChangeCounter / dataVectorLength;
}

void KMeansGPU2Solver::findNearestClusters()
{
	int blockSize = 1024;
	int blocks = dataVectorLength / 1024 + 1;
	findNearestClustersKernel2<<<blocks, blockSize>>>(dataVectors, numberOfDimensions, centroidMemberships, thrust_MembershipChangeVector, newCentroidVectors, thrust_centroidMembershipCounts, centroidVectors, centroidVectorLength, dataVectorLength);
	cudaCheck2(cudaGetLastError(), CGLE_ERR);
	cudaCheck2(cudaDeviceSynchronize(), CDS_ERR);
}

void KMeansGPU2Solver::averageNewClusters()
{
	for (int i = 0; i < centroidVectorLength; i++) {
		int centroidMembershipCount = thrust::count_if(thrust_centroidMembershipCounts, thrust_centroidMembershipCounts + dataVectorLength, is_equal_to(i));
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = newCentroidVectors[i * numberOfDimensions + j] / centroidMembershipCount;
	}	
}

void KMeansGPU2Solver::clearSolver()
{
	cudaCheck2(cudaDeviceReset(), CDR_ERR);
}
