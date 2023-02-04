#pragma once
#include <cuda_runtime_api.h>

template <int numberOfDimensions> class KMeansGPU2Solver {

public:
	float* centroidVectors;

private:

	float* dataVectors;
	int* centroidMemberships;
	int* centroidKeys;

	int dataVectorLength;
	int centroidVectorLength;

	int* membershipChangeVector;
	float threshold;
	int limit;

	float* copiedData;
	float* copiedMemberships;
	float* centroids;
	int* keys;

public:
	void initSolver(float* dataVectors, int dataVectorLength, int centroidCount, float threshold, int limit);
	void solve();
	void clearSolver();

private:
	float getMembershipChangeFraction();
	void findNearestClusters();
	void averageNewClusters();
	void clearVariables();
	void cudaCheck(cudaError_t status, char* message);
};