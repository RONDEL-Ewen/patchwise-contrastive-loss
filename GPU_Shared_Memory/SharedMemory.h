/**
 * SharedMemory.h
 * Header file for shared memory on GPU.
 * Declares interfaces for kernels and functions that extract features.
 */

#pragma once

#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>

void extractFeaturesShared(
    unsigned char* refPatch,
    unsigned char* posPatch,
    unsigned char* negPatches,
    float* embeddings,
    int numPatches,
    int patchWidth,
    int patchHeight,
    int numChannels
);

#endif // SHARED_MEMORY_H