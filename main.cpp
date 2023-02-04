#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#include "kMeansCPU.h"
#include "kMeansGPU1.cuh"
#include "kMeansGPU2.cuh"

#define N 10000000
#define K 10
#define D 2
#define EPS 1 / N

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
    KMeansGPU2Solver kMeansGPU2Solver;
    kMeansGPU2Solver.initSolver(dataVectors, dataVectorLength, numberOfDimensions, centroidVectorLength, threshold);

    printf("Initial Centroids\n-----------------------------------\n");
    printSolution(dataVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");

    /*auto startGPU = std::chrono::high_resolution_clock::now();
    kMeansGPU1Solver.solve();
    auto stopGPU = std::chrono::high_resolution_clock::now();

    printf("Warm up GPU\n-----------------------------------\nComputation time: %dms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(stopGPU - startGPU));
    printSolution(kMeansGPU1Solver.centroidVectors, centroidVectorLength, numberOfDimensions);
    kMeansGPU1Solver.initSolver(dataVectors, dataVectorLength, numberOfDimensions, centroidVectorLength, threshold);*/

    /*auto startCPU = std::chrono::high_resolution_clock::now();
    kMeansCPUSolver.solve();
    auto stopCPU = std::chrono::high_resolution_clock::now();


    printf("CPU Solution\n-----------------------------------\nComputation time: %dms\n", 
        std::chrono::duration_cast<std::chrono::milliseconds>(stopCPU - startCPU));
    printSolution(kMeansCPUSolver.centroidVectors, centroidVectorLength, numberOfDimensions);
    printf("\n\n");*/

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

void clearDataVectors(float* dataVectors)
{
}
