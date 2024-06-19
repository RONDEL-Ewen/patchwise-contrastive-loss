/**
 * NaiveParallelization.h
 * Header file for naive parallelization on GPU.
 * Declares interfaces for kernels and functions that extract patches, extract features and compute loss.
 */

#pragma once

#ifndef NAIVE_PARALLELIZATION_H
#define NAIVE_PARALLELIZATION_H

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>

void extractPatchesNaivePara(
    const unsigned char* inputImage,
    const unsigned char* outputImage,
    int imageWidth,
    int imageHeight,
    int patchSize,
    int numChannels,
    int numNegativePatches,
    unsigned char* refPatch,
    unsigned char* posPatch,
    unsigned char* negPatches
);

void extractFeaturesNaivePara(
    unsigned char* refPatch,
    unsigned char* posPatch,
    unsigned char* negPatches,
    float* embeddings,
    int numPatches,
    int patchWidth,
    int patchHeight,
    int numChannels
);

void computeLossNaivePara(
    float *embeddings,
    float &loss,
    int numEmbeddings,
    int embLength,
    float margin
);

#endif // NAIVE_PARALLELIZATION_H