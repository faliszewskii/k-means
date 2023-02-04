#pragma once

template <int numberOfDimensions> class KMeansCPUSolver {

public:
	float* centroidVectors;

private:

	float* dataVectors;
	float* newCentroidVectors;
	int* centroidMemberships;
	int* centroidMembershipCounts;

	int dataVectorLength;
	int centroidVectorLength;

	int membershipChangeCounter;
	float threshold;
	int limit;

public:
	void initSolver(float* dataVectors, int dataVectorLength, int centroidCount, float threshold, int limit);
	void solve();
	void clearSolver();

private:
	void findNearestClusters();
	void averageNewClusters();
	int findNearestClusterFor(float* vector);
	void clearVariables();
};