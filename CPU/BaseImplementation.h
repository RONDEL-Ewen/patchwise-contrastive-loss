/**
 * BaseImplementation.h
 * Header file for CPU implementation.
 * Declares interfaces for functions that extract patches, extract features and compute loss.
 */

#pragma once

#ifndef BASE_IMPLEMENTATION_H
#define BASE_IMPLEMENTATION_H

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>

void extractPatches(
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

void extractFeatures(
    unsigned char* refPatch,
    unsigned char* posPatch,
    unsigned char* negPatches,
    float* embeddings,
    int numPatches,
    int patchWidth,
    int patchHeight,
    int numChannels
);

void computeLoss(
    float *embeddings,
    float &loss,
    int numEmbeddings,
    int embLength,
    float margin
);

#endif // BASE_IMPLEMENTATION_H