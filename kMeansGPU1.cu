#include "kMeansGPU1.cuh"
#include <limits>


void KMeansGPU1Solver::solve()
{
	float membershipChangeFraction;

	do {
		clearVariables();
		findNearestClusters();
		averageNewClusters();
		membershipChangeFraction = membershipChangeCounter / dataVectorLength;
	} while (membershipChangeFraction > threshold);
}


void KMeansGPU1Solver::initSolver(float** dataVectors, int dataVectorLength, int numberOfDimensions, int centroidVectorLength, float threshold)
{
	this->dataVectors = dataVectors;
	this->dataVectorLength = dataVectorLength;
	this->numberOfDimensions = numberOfDimensions;
	this->centroidVectorLength = centroidVectorLength;
	this->threshold = threshold;
	this->membershipChangeCounter = 0;

	currentDistances = new float[dataVectorLength];
	centroidMemberships = new int[dataVectorLength];
	centroidMembershipCounts = new int[centroidVectorLength];

	centroidVectors = new float* [centroidVectorLength];
	newCentroidVectors = new float* [centroidVectorLength];
	for (int i = 0; i < centroidVectorLength; i++) {
		centroidVectors[i] = new float[numberOfDimensions];
		newCentroidVectors[i] = new float[numberOfDimensions];
	}

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i][j] = dataVectors[i][j];
}

void KMeansGPU1Solver::clearVariables()
{
	membershipChangeCounter = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			newCentroidVectors[i][j] = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		centroidMembershipCounts[i] = 0;
}

void KMeansGPU1Solver::findNearestClusters()
{
	for (int i = 0; i < dataVectorLength; i++) {
		int index = findNearestClusterFor(dataVectors[i]);
		if (centroidMemberships[i] != index) {
			membershipChangeCounter++;
			centroidMemberships[i] = index;
		}
		for (int j = 0; j < numberOfDimensions; j++)
			newCentroidVectors[index][j] += dataVectors[i][j];
		centroidMembershipCounts[index]++;
	}
}

void KMeansGPU1Solver::averageNewClusters()
{
	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i][j] = newCentroidVectors[i][j] / centroidMembershipCounts[i];
}

int KMeansGPU1Solver::findNearestClusterFor(float* vector)
{
	int minDistanceIndex = 0;
	float minDistanceSquared = std::numeric_limits<float>::max();
	float distanceSquared = 0;

	for (int i = 0; i < centroidVectorLength; i++) {
		distanceSquared = 0;
		for (int j = 0; j < numberOfDimensions; j++)
			distanceSquared += powf(vector[j] - centroidVectors[i][j], 2);
		if (distanceSquared < minDistanceSquared) {
			minDistanceSquared = distanceSquared;
			minDistanceIndex = i;
		}
	}

	return minDistanceIndex;
}

void KMeansGPU1Solver::clearSolver()
{
}
