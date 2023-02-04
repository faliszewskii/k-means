#pragma once
#include <thrust/device_ptr.h>
#include <vector>

class KMeansGPU2Solver {

public:
	float* centroidVectors;

private:

	float* dataVectors;
	int* centroidMemberships;
	int* centroidKeys;

	int dataVectorLength;
	int numberOfDimensions;
	int centroidVectorLength;

	int* membershipChangeVector;
	float threshold;
	int limit;

	float* copiedData;
	float* copiedMemberships;
	float* centroids;
	int* keys;

public:
	void initSolver(float* dataVectors, int dataVectorLength, int numberOfDimensions, int centroidCount, float threshold, int limit);
	void solve();
	void clearSolver();

private:
	float getMembershipChangeFraction();
	void findNearestClusters();
	void averageNewClusters();
	void clearVariables();
};