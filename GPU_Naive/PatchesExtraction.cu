/**
 * PatchesExtraction.cu
 * Contains CUDA kernels and functions for extracting reference, positive,
 * and negative patches from input and output images.
 * This module is crucial for the initial phase of the Patchwise Contrastive Loss computation.
 */

#include "NaiveParallelization.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK_CUDA_ERROR(call) {                                           \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

// ==================== Patches Coordinates ====================

__global__ void generateRandomCoordinatesNaivePara(int* coords, int width, int height, int patchSize, int numPatches, unsigned long long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);

    if (idx < numPatches) {
        // Generate random x and y coordinates for placing the patches
        coords[2 * idx] = curand(&state) % (width - patchSize);  // x coordinate
        coords[2 * idx + 1] = curand(&state) % (height - patchSize);  // y coordinate
    }
}

// ==================== Extract 1 Patch ====================

__global__ void extractPatchKernelNaivePara(const unsigned char* image, unsigned char* patch, int startX, int startY, int imageWidth, int patchSize, int numChannels) {
    
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < patchSize && y < patchSize) {
        int imageIndex = ((startY + y) * imageWidth + (startX + x)) * numChannels;
        int patchIndex = (y * patchSize + x) * numChannels;
        for (int channel = 0; channel < numChannels; channel++) {
            patch[patchIndex + channel] = image[imageIndex + channel];
        }
    }
}

// ==================== extractPatches ====================

void extractPatchesNaivePara(const unsigned char* inputImage, const unsigned char* outputImage, int imageWidth, int imageHeight, int patchSize, int numChannels, int numNegativePatches, unsigned char* refPatch, unsigned char* posPatch, unsigned char* negPatches) {

    int numPatches = numNegativePatches + 2;

    // Allocate memory for coordinates on the host and device
    int* hostCoords = (int*)malloc(numPatches * 2 * sizeof(int));
    int* devCoords;
    CHECK_CUDA_ERROR(cudaMalloc(&devCoords, numPatches * 2 * sizeof(int)));

    // Generate coordinates sequentially
    for (int i = 0; i < numPatches; i++) {
        generateRandomCoordinatesNaivePara<<<1, 1>>>(devCoords + i * 2, imageWidth, imageHeight, patchSize, 1, time(NULL) + i);
        CHECK_CUDA_ERROR(cudaMemcpy(&hostCoords[i * 2], devCoords + i * 2, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // Allocate memory for images on device
    unsigned char *devInputImage, *devOutputImage;
    size_t imageSize = (size_t)imageWidth * imageHeight * numChannels;
    CHECK_CUDA_ERROR(cudaMalloc(&devInputImage, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&devOutputImage, imageSize));
    CHECK_CUDA_ERROR(cudaMemcpy(devInputImage, inputImage, imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(devOutputImage, outputImage, imageSize, cudaMemcpyHostToDevice));

    // Allocate memory for patches on device
    unsigned char *devRefPatch, *devPosPatch, *devNegPatches;
    size_t patchSizeBytes = patchSize * patchSize * numChannels * sizeof(unsigned char);
    CHECK_CUDA_ERROR(cudaMalloc(&devRefPatch, patchSizeBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&devPosPatch, patchSizeBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&devNegPatches, numNegativePatches * patchSizeBytes));

    // Allocate streams
    cudaStream_t* streams = (cudaStream_t*)malloc(numPatches * sizeof(cudaStream_t));
    if (streams == NULL) {
        // Error handling
        fprintf(stderr, "Failed to allocate memory for streams\n");
        exit(EXIT_FAILURE);
    }

    // Create streams
    for (int i = 0; i < numPatches; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    // Configure dimensions for kernel launch
    dim3 blockDims(16, 16);
    dim3 gridDims((patchSize + blockDims.x - 1) / blockDims.x, (patchSize + blockDims.y - 1) / blockDims.y);

    // Extract reference patch
    extractPatchKernelNaivePara<<<gridDims, blockDims, 0, streams[0]>>>(devInputImage, devRefPatch, hostCoords[0], hostCoords[1], imageWidth, patchSize, numChannels);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Extract positive patch
    extractPatchKernelNaivePara<<<gridDims, blockDims, 0, streams[1]>>>(devOutputImage, devPosPatch, hostCoords[0], hostCoords[1], imageWidth, patchSize, numChannels);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Extract negative patches
    for (int i = 0; i < numNegativePatches; i++) {
        //printf("+1");
        int offset = 2 * (i + 1);  // Offset for coordinates
        extractPatchKernelNaivePara<<<gridDims, blockDims, 0, streams[i + 2]>>>(devOutputImage, devNegPatches + i * patchSizeBytes,
                                                                    hostCoords[offset], hostCoords[offset + 1], imageWidth, patchSize, numChannels);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    // Synchronize all streams
    for (int i = 0; i < numPatches; i++) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
    }

    // Copy patches to host memory
    CHECK_CUDA_ERROR(cudaMemcpy(refPatch, devRefPatch, patchSizeBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(posPatch, devPosPatch, patchSizeBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(negPatches, devNegPatches, numNegativePatches * patchSizeBytes, cudaMemcpyDeviceToHost));

    // Destroy streams
    for (int i = 0; i < numPatches; i++) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(devInputImage));
    CHECK_CUDA_ERROR(cudaFree(devOutputImage));
    CHECK_CUDA_ERROR(cudaFree(devCoords));
    CHECK_CUDA_ERROR(cudaFree(devRefPatch));
    CHECK_CUDA_ERROR(cudaFree(devPosPatch));
    CHECK_CUDA_ERROR(cudaFree(devNegPatches));

    // Free the allocated host memory and stream memory
    free(hostCoords);
    free(streams);
}