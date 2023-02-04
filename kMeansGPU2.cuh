#pragma once
#include <thrust/device_ptr.h>

class KMeansGPU2Solver {

public:
	float* centroidVectors;

private:

	float* dataVectors;
	float* newCentroidVectors;
	int* centroidMemberships;
	int* centroidMembershipCounts;

	int dataVectorLength;
	int numberOfDimensions;
	int centroidVectorLength;

	int* membershipChangeVector;
	float threshold;
	int limit;

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