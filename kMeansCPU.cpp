#include "kMeansCPU.h"
#include <limits>

template <int numberOfDimensions> void KMeansCPUSolver<numberOfDimensions>::solve()
{
	float membershipChangeFraction;
	int iteration = 0;
	
	do {
		clearVariables();
		findNearestClusters();
		averageNewClusters();
		membershipChangeFraction = (float)membershipChangeCounter / dataVectorLength;
	} while (iteration < limit && membershipChangeFraction > threshold);
}


template <int numberOfDimensions> void KMeansCPUSolver<numberOfDimensions>::initSolver(float* dataVectors, int dataVectorLength, int centroidCount, float threshold, int limit)
{
	this->dataVectors = dataVectors;
	this->threshold = threshold;
	this->membershipChangeCounter = 0;
	this->dataVectorLength = dataVectorLength;
	this->centroidVectorLength = centroidCount;
	this->limit = limit;

	centroidMemberships = new int[dataVectorLength];
	centroidMembershipCounts = new int[centroidVectorLength];

	centroidVectors = new float[centroidVectorLength * numberOfDimensions];
	newCentroidVectors = new float[centroidVectorLength * numberOfDimensions];

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = dataVectors[i * numberOfDimensions + j];
}

template <int numberOfDimensions> void KMeansCPUSolver<numberOfDimensions>::clearVariables()
{
	membershipChangeCounter = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			newCentroidVectors[i * numberOfDimensions + j] = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		centroidMembershipCounts[i] = 0;
}

template <int numberOfDimensions> void KMeansCPUSolver<numberOfDimensions>::findNearestClusters()
{
	for (int i = 0; i < dataVectorLength; i++) {
		int index = findNearestClusterFor(&dataVectors[i * numberOfDimensions]);
		if (centroidMemberships[i] != index) {
			membershipChangeCounter++;
			centroidMemberships[i] = index;
		}
		for (int j = 0; j < numberOfDimensions; j++)
			newCentroidVectors[index * numberOfDimensions + j] += dataVectors[i * numberOfDimensions + j];
		centroidMembershipCounts[index]++;
	}
}

template <int numberOfDimensions> void KMeansCPUSolver<numberOfDimensions>::averageNewClusters()
{
	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = newCentroidVectors[i * numberOfDimensions + j] / centroidMembershipCounts[i];
}

template <int numberOfDimensions> int KMeansCPUSolver<numberOfDimensions>::findNearestClusterFor(float* vector)
{
	int minDistanceIndex = 0;
	float minDistanceSquared = std::numeric_limits<float>::max();
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

template <int numberOfDimensions> void KMeansCPUSolver<numberOfDimensions>::clearSolver()
{
}


