#include <stdio.h>
#include <stdlib.h>

#include "kMeansCPU.h"

#define N 10000
#define K 5
#define D 3
#define EPS 0.001 * N + 1

float** generateDataVectors(int dataVectorLength, int numberOfDimensions);
void printSolution(float** solution, int dimX, int dimY);
void clearDataVectors(float** dataVectors);

int main()
{
    const int dataVectorLength = N;
    const int numberOfDimensions = D;
    const int centroidVectorLength = K;
    const float threshold = EPS;

    float** dataVectors = generateDataVectors(dataVectorLength, numberOfDimensions);

    KMeansCPUSolver kMeansCPUSolver;
    kMeansCPUSolver.initSolver(dataVectors, dataVectorLength, numberOfDimensions, centroidVectorLength, threshold);
    kMeansCPUSolver.solve();

    KMeansGPU1Solver kMeansGPU1Solver;
    kMeansGPU1Solver.initSolver(dataVectors, dataVectorLength, numberOfDimensions, centroidVectorLength, threshold);
    kMeansGPU1Solver.solve();

    printf("Initial Centroids\n-----------------------------------\n");
    printSolution(dataVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    printf("CPU Solution\n-----------------------------------\n");
    printSolution(kMeansCPUSolver.centroidVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    clearDataVectors(dataVectors);
    return 0;
}

float** generateDataVectors(int dataVectorLength, int numberOfDimensions) 
{
    srand(0);

    float** dataVectors = new float* [dataVectorLength];
    for (int i = 0; i < dataVectorLength; i++) {
        dataVectors[i] = new float[numberOfDimensions];
        for (int j = 0; j < numberOfDimensions; j++)
            dataVectors[i][j] = rand() % 100000 / 100000.f;
    }
    return dataVectors;
}

void printSolution(float** solution, int dimX, int dimY)
{
    for (int i = 0; i < dimX; i++) {
        printf("%d:", i);
        for (int j = 0; j < dimY; j++)
            printf("\t%06f", solution[i][j]);
        printf("\n");
    }
}

void clearDataVectors(float** dataVectors)
{
}
