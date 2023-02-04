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
	thrust::device_ptr<int> thrust_centroidMembershipCounts;

	int dataVectorLength;
	int numberOfDimensions;
	int centroidVectorLength;

	thrust::device_ptr<int> thrust_MembershipChangeVector;
	float threshold;

public:
	void initSolver(float* dataVectors, int dataVectorLength, int numberOfDimensions, int centroidCount, float threshold);
	void solve();
	void clearSolver();

private:
	float getMembershipChangeFraction();
	void findNearestClusters();
	void averageNewClusters();
	void clearVariables();
};