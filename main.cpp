#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kMeansCPU.h"
#include "kMeansGPU1.cuh"

#define N 100000000
#define K 10
#define D 3
#define EPS 0.0000001 * N + 1

float* generateDataVectors(int dataVectorLength, int numberOfDimensions);
void printSolution(float* solution, int dimX, int dimY);
void clearDataVectors(float* dataVectors);

int main()
{
    const int dataVectorLength = N;
    const int numberOfDimensions = D;
    const int centroidVectorLength = K;
    const float threshold = EPS;

    float* dataVectors = generateDataVectors(dataVectorLength, numberOfDimensions);

    KMeansCPUSolver kMeansCPUSolver;
    kMeansCPUSolver.initSolver(dataVectors, dataVectorLength, numberOfDimensions, centroidVectorLength, threshold);
    KMeansGPU1Solver kMeansGPU1Solver;
    kMeansGPU1Solver.initSolver(dataVectors, dataVectorLength, numberOfDimensions, centroidVectorLength, threshold);

    printf("Initial Centroids\n-----------------------------------\n");
    printSolution(dataVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    kMeansCPUSolver.solve();

    printf("CPU Solution\n-----------------------------------\n");
    printSolution(kMeansCPUSolver.centroidVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    kMeansGPU1Solver.solve();

    printf("GPU Solution No. 1\n-----------------------------------\n");
    printSolution(kMeansGPU1Solver.centroidVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    clearDataVectors(dataVectors);
    kMeansCPUSolver.clearSolver();
    kMeansGPU1Solver.clearSolver();
    return 0;
}

float* generateDataVectors(int dataVectorLength, int numberOfDimensions)
{
    srand(time(NULL));

    float* dataVectors = new float[dataVectorLength * numberOfDimensions];
    for (int i = 0; i < dataVectorLength; i++) {
        for (int j = 0; j < numberOfDimensions; j++)
            dataVectors[i * numberOfDimensions + j] = rand() % 100000 / 100000.f;
    }
    return dataVectors;
}

void printSolution(float* solution, int dimX, int dimY)
{
    for (int i = 0; i < dimX; i++) {
        printf("%d:", i);
        for (int j = 0; j < dimY; j++)
            printf("\t%06f", solution[i * dimY + j]);
        printf("\n");
    }
}

void clearDataVectors(float* dataVectors)
{
}
