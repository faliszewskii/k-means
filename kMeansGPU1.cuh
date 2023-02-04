#pragma once
#include <cuda_runtime_api.h>

template <int numberOfDimensions> class KMeansGPU1Solver {

public:
	float* centroidVectors;

private:


	float* dataVectors;
	float* newCentroidVectors;
	int* centroidMemberships;
	int* centroidMembershipCounts;

	int dataVectorLength;
	int centroidVectorLength;

	int* membershipChangeCounter;
	float threshold;
	int limit;

public:
	void initSolver(float* dataVectors, int dataVectorLength, int centroidCount, float threshold, int limit);
	void solve();
	void clearSolver();

private:
	void findNearestClusters();
	void averageNewClusters();
	void clearVariables();
	void cudaCheck(cudaError_t status, char* message);
};