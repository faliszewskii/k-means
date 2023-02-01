#pragma once

class KMeansGPU1Solver {

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

	int* membershipChangeCounter;
	float threshold;

public:
	void initSolver(float* dataVectors, int dataVectorLength, int numberOfDimensions, int centroidCount, float threshold);
	void solve();
	void clearSolver();

private:
	void findNearestClusters();
	void averageNewClusters();
	int findNearestClusterFor(float* vector);
	void clearVariables();
};