#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#include "kMeansCPU.h"
#include "kMeansGPU1.cuh"
#include "kMeansGPU2.cuh"

#define N 2000000
#define K 8
#define D 3
#define EPS 10 / N
#define LIM 100


float* generateDataVectors(int dataVectorLength, int numberOfDimensions);
void printSolution(float* solution, int dimX, int dimY);
void printSolution(float** solution, int dimX, int dimY);
void clearDataVectors(float* dataVectors);

int main()
{
    const int dataVectorLength = N;
    const int centroidVectorLength = K;
    const int numberOfDimensions = D;
    const float threshold = EPS;
    const int limit = LIM;

    float* dataVectors = generateDataVectors(dataVectorLength, numberOfDimensions);

    KMeansCPUSolver<numberOfDimensions> kMeansCPUSolver;
    kMeansCPUSolver.initSolver(dataVectors, dataVectorLength, centroidVectorLength, threshold, limit);
    KMeansGPU1Solver<numberOfDimensions> kMeansGPU1Solver;
    kMeansGPU1Solver.initSolver(dataVectors, dataVectorLength, centroidVectorLength, threshold, limit);
    KMeansGPU2Solver<numberOfDimensions> kMeansGPU2Solver;
    kMeansGPU2Solver.initSolver(dataVectors, dataVectorLength, centroidVectorLength, threshold, limit);

    printf("Initial Centroids\n-----------------------------------\n");
    printSolution(dataVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    auto startCPU = std::chrono::high_resolution_clock::now();
    kMeansCPUSolver.solve();
    auto stopCPU = std::chrono::high_resolution_clock::now();


    printf("CPU Solution\n-----------------------------------\nComputation time: %dms\n", 
        std::chrono::duration_cast<std::chrono::milliseconds>(stopCPU - startCPU));
    printSolution(kMeansCPUSolver.centroidVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    auto startGPU1 = std::chrono::high_resolution_clock::now();
    kMeansGPU1Solver.solve();
    auto stopGPU1 = std::chrono::high_resolution_clock::now();

    printf("GPU Solution No. 1\n-----------------------------------\nComputation time: %dms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(stopGPU1 - startGPU1));
    printSolution(kMeansGPU1Solver.centroidVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    auto startGPU2 = std::chrono::high_resolution_clock::now();
    kMeansGPU2Solver.solve();
    auto stopGPU2 = std::chrono::high_resolution_clock::now();

    printf("GPU Solution No. 2\n-----------------------------------\nComputation time: %dms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(stopGPU2 - startGPU2));
    printSolution(kMeansGPU2Solver.centroidVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    clearDataVectors(dataVectors);
    kMeansCPUSolver.clearSolver();
    kMeansGPU1Solver.clearSolver();
    kMeansGPU2Solver.clearSolver();
    return 0;
}

float* generateDataVectors(int dataVectorLength, int numberOfDimensions)
{
    srand(time(NULL));

    float* dataVectors = new float[dataVectorLength * numberOfDimensions];
    for (int i = 0; i < dataVectorLength; i++) {
        for (int j = 0; j < numberOfDimensions; j++)
            dataVectors[i * numberOfDimensions + j] = rand() % 10000 / 10000.f;
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

void printSolution(float** solution, int dimX, int dimY)
{
    for (int i = 0; i < dimX; i++) {
        printf("%d:", i);
        for (int j = 0; j < dimY; j++)
            printf("\t%06f", solution[i][j]);
        printf("\n");
    }
}

void clearDataVectors(float* dataVectors)
{
}
