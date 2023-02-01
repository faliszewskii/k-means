#include "kMeansCPU.h"
#include <limits>


void KMeansCPUSolver::solve()
{
	float membershipChangeFraction;
	
	do {
		clearVariables();
		findNearestClusters();
		averageNewClusters();
		membershipChangeFraction = (float)membershipChangeCounter / dataVectorLength;
	} while (membershipChangeFraction > threshold);
}


void KMeansCPUSolver::initSolver(float* dataVectors, int dataVectorLength, int numberOfDimensions, int centroidVectorLength, float threshold)
{
	this->dataVectors = dataVectors;
	this->dataVectorLength = dataVectorLength;
	this->numberOfDimensions = numberOfDimensions;
	this->centroidVectorLength = centroidVectorLength;
	this->threshold = threshold;
	this->membershipChangeCounter = 0;

	centroidMemberships = new int[dataVectorLength];
	centroidMembershipCounts = new int[centroidVectorLength];

	centroidVectors = new float[centroidVectorLength * numberOfDimensions];
	newCentroidVectors = new float[centroidVectorLength * numberOfDimensions];

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = dataVectors[i * numberOfDimensions + j];
}

void KMeansCPUSolver::clearVariables()
{
	membershipChangeCounter = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			newCentroidVectors[i * numberOfDimensions + j] = 0;

	for (int i = 0; i < centroidVectorLength; i++)
		centroidMembershipCounts[i] = 0;
}

void KMeansCPUSolver::findNearestClusters()
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

void KMeansCPUSolver::averageNewClusters()
{
	for (int i = 0; i < centroidVectorLength; i++)
		for (int j = 0; j < numberOfDimensions; j++)
			centroidVectors[i * numberOfDimensions + j] = newCentroidVectors[i * numberOfDimensions + j] / centroidMembershipCounts[i];
}

int KMeansCPUSolver::findNearestClusterFor(float* vector)
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

void KMeansCPUSolver::clearSolver()
{
}
