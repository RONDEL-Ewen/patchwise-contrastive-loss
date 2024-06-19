/**
 * PatchesFeatures.cu
 * Defines CUDA kernels and functions for computing various image features such as RGB averages,
 * color histograms, texture histograms, oriented gradient histograms, and Hu moments.
 * These features are extracted from the image patches and are used to form embeddings.
 */

#include "NaiveParallelization.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

#define CH_BINS 256
#define CHANNELS 3
#define GH_BINS 36
#define PI 3.14159265358979323846
#define TH_BINS 256
#define FEATURE_LENGTH (3 + CH_BINS * CHANNELS + GH_BINS + TH_BINS + 7)

#define CHECK_CUDA_ERROR(call) {                                           \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

// Kernels declaration
__global__ void calculateRGBMeansKernelNaivePara(unsigned char* patch, float* rgbMeans, int totalPixels);
__global__ void calculateColorHistogramKernelNaivePara(unsigned char* patch, float* histogram, int totalPixels);
__global__ void calculateOrientedGradientHistogramKernelNaivePara(unsigned char* patch, float* histogram, int totalPixels, int patchWidth, int patchHeight, int numChannels);
__global__ void calculateTextureHistogramKernelNaivePara(unsigned char* patch, float* histogram, int totalPixels, int patchWidth, int patchHeight, int numChannels);
__global__ void calculateHuMomentsKernelNaivePara(unsigned char* patch, float* huMoments, int patchWidth, int patchHeight, int numChannels);

// ==================== extractFeatures ====================

void extractFeaturesNaivePara(unsigned char* refPatch, unsigned char* posPatch, unsigned char* negPatches, float* embeddings, int numPatches, int patchWidth, int patchHeight, int numChannels) {
    int patchSize = patchWidth * patchHeight * numChannels;
    int totalPixels = patchWidth * patchHeight;

    unsigned char* d_patches;
    float* d_embeddings;
    cudaStream_t* streams = new cudaStream_t[numPatches];

    // Allocate and transfer all patches at once
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_patches, patchSize * numPatches * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_patches, refPatch, patchSize * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_patches + patchSize, posPatch, patchSize * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_patches + 2 * patchSize, negPatches, patchSize * (numPatches - 2) * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Allocate space for all embeddings
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_embeddings, numPatches * FEATURE_LENGTH * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_embeddings, 0, numPatches * FEATURE_LENGTH * sizeof(float)));

    // Create and initialize CUDA streams
    for (int i = 0; i < numPatches; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((patchWidth + blockSize.x - 1) / blockSize.x, (patchHeight + blockSize.y - 1) / blockSize.y);

    // Launch kernels for each patch in separate streams
    for (int p = 0; p < numPatches; p++) {
        unsigned char* currentPatch = d_patches + p * patchSize;
        float* currentEmbeddings = d_embeddings + p * FEATURE_LENGTH;

        calculateRGBMeansKernelNaivePara<<<gridSize, blockSize, 0, streams[p]>>>(currentPatch, currentEmbeddings, totalPixels);
        calculateColorHistogramKernelNaivePara<<<gridSize, blockSize, 0, streams[p]>>>(currentPatch, currentEmbeddings + 3, totalPixels);
        calculateOrientedGradientHistogramKernelNaivePara<<<gridSize, blockSize, 0, streams[p]>>>(currentPatch, currentEmbeddings + 3 + CH_BINS * CHANNELS, totalPixels, patchWidth, patchHeight, numChannels);
        calculateTextureHistogramKernelNaivePara<<<gridSize, blockSize, 0, streams[p]>>>(currentPatch, currentEmbeddings + 3 + CH_BINS * CHANNELS + GH_BINS, totalPixels, patchWidth, patchHeight, numChannels);
        calculateHuMomentsKernelNaivePara<<<gridSize, blockSize, 0, streams[p]>>>(currentPatch, currentEmbeddings + 3 + CH_BINS * CHANNELS + GH_BINS + TH_BINS, patchWidth, patchHeight, numChannels);
    }

    // Wait for all streams to finish
    for (int i = 0; i < numPatches; i++) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }

    // Copy embeddings back to host
    CHECK_CUDA_ERROR(cudaMemcpy(embeddings, d_embeddings, numPatches * FEATURE_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_patches);
    cudaFree(d_embeddings);
    delete[] streams;
}

// ==================== Mean RGB ====================

__global__ void calculateRGBMeansKernelNaivePara(unsigned char* patch, float* rgbMeans, int totalPixels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

    for (int i = index; i < totalPixels; i += stride) {
        int pixelIndex = i * CHANNELS;
        redSum += patch[pixelIndex];
        greenSum += patch[pixelIndex + 1];
        blueSum += patch[pixelIndex + 2];
    }

    atomicAdd(&rgbMeans[0], redSum);
    atomicAdd(&rgbMeans[1], greenSum);
    atomicAdd(&rgbMeans[2], blueSum);

    __syncthreads();

    if (threadIdx.x == 0) {
        rgbMeans[0] /= (totalPixels * 255.0);
        rgbMeans[1] /= (totalPixels * 255.0);
        rgbMeans[2] /= (totalPixels * 255.0);
    }
}

// ==================== Color Histogram ====================

__global__ void calculateColorHistogramKernelNaivePara(unsigned char* patch, float* histogram, int totalPixels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < totalPixels; i += stride) {
        int pixelIndex = i * CHANNELS;
        atomicAdd(&histogram[patch[pixelIndex]], 1);
        atomicAdd(&histogram[256 + patch[pixelIndex + 1]], 1);
        atomicAdd(&histogram[512 + patch[pixelIndex + 2]], 1);
    }

    __syncthreads();
    
    if (index < CH_BINS * CHANNELS) {
        histogram[index] /= totalPixels;
    }
}

// ==================== Oriented Gradients Histogram ====================

__global__ void calculateOrientedGradientHistogramKernelNaivePara(unsigned char* patch, float* histogram, int totalPixels, int patchWidth, int patchHeight, int numChannels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < patchWidth - 1 && y > 0 && y < patchHeight - 1) {
        int pixelIndex = (y * patchWidth + x) * numChannels;
        float Gx = -1 * patch[pixelIndex - numChannels - patchWidth*numChannels]
                   + 1 * patch[pixelIndex + numChannels - patchWidth*numChannels]
                   - 2 * patch[pixelIndex - numChannels]
                   + 2 * patch[pixelIndex + numChannels]
                   - 1 * patch[pixelIndex - numChannels + patchWidth*numChannels]
                   + 1 * patch[pixelIndex + numChannels + patchWidth*numChannels];
        float Gy = -1 * patch[pixelIndex - patchWidth*numChannels - numChannels]
                   - 2 * patch[pixelIndex - patchWidth*numChannels]
                   - 1 * patch[pixelIndex - patchWidth*numChannels + numChannels]
                   + 1 * patch[pixelIndex + patchWidth*numChannels - numChannels]
                   + 2 * patch[pixelIndex + patchWidth*numChannels]
                   + 1 * patch[pixelIndex + patchWidth*numChannels + numChannels];
        float angle = atan2f(Gy, Gx) * 180.0f / PI;
        if (angle < 0) angle += 360.0f;
        int bin = (int)(angle / (360.0f / GH_BINS));
        atomicAdd(&histogram[bin], 1);
    }

    __syncthreads();
    if (x < GH_BINS) {
        histogram[x] /= totalPixels;
    }
}

// ==================== Texture Histogram ====================

__global__ void calculateTextureHistogramKernelNaivePara(unsigned char* patch, float* histogram, int totalPixels, int patchWidth, int patchHeight, int numChannels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < patchWidth - 1 && y > 0 && y < patchHeight - 1) {
        unsigned char center = patch[(y * patchWidth + x) * numChannels];
        unsigned char code = 0;
        int dx[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
        int dy[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            unsigned char neighbor = patch[(ny * patchWidth + nx) * numChannels];
            if (neighbor > center) code |= (1 << i);
        }
        atomicAdd(&histogram[code], 1);
    }

    __syncthreads();
    if (x < TH_BINS) {
        histogram[x] /= totalPixels;
    }
}

// ==================== Hu Moments ====================

__global__ void calculateHuMomentsKernelNaivePara(unsigned char* patch, float* huMoments, int patchWidth, int patchHeight, int numChannels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * patchWidth + x;

    if (x < patchWidth && y < patchHeight) {
        unsigned char grayValue;
        if (numChannels == 1) {
            grayValue = patch[index];  // Already grayscale
        } else {
            // Convert to grayscale using standard luminosity formula
            grayValue = 0.299f * patch[index * numChannels] + 0.587f * patch[index * numChannels + 1] + 0.114f * patch[index * numChannels + 2];
        }

        // Compute raw moments
        atomicAdd(&huMoments[0], grayValue);  // m00
        atomicAdd(&huMoments[1], grayValue * x);  // m10
        atomicAdd(&huMoments[2], grayValue * y);  // m01
        atomicAdd(&huMoments[3], grayValue * x * y);  // m11
        atomicAdd(&huMoments[4], grayValue * x * x);  // m20
        atomicAdd(&huMoments[5], grayValue * y * y);  // m02
        atomicAdd(&huMoments[6], grayValue * x * x * x);  // m30
        atomicAdd(&huMoments[7], grayValue * x * y * y);  // m12
        atomicAdd(&huMoments[8], grayValue * x * x * y);  // m21
        atomicAdd(&huMoments[9], grayValue * y * y * y);  // m03
    }
    
    __syncthreads();

    // Only one thread calculates the central moments and Hu moments
    if (x == 0 && y == 0) {
        float m00 = huMoments[0];
        float m10 = huMoments[1];
        float m01 = huMoments[2];
        float x_bar = m10 / m00;
        float y_bar = m01 / m00;

        float mu20 = huMoments[4] - x_bar * m10;
        float mu02 = huMoments[5] - y_bar * m01;
        float mu11 = huMoments[3] - x_bar * m01;
        float mu30 = huMoments[6] - 3 * x_bar * huMoments[4] + 2 * x_bar * x_bar * m10;
        float mu12 = huMoments[7] - x_bar * huMoments[5] - 2 * y_bar * huMoments[3] + 2 * y_bar * y_bar * m01;
        float mu21 = huMoments[8] - y_bar * huMoments[4] - 2 * x_bar * huMoments[3] + 2 * x_bar * x_bar * m01;
        float mu03 = huMoments[9] - 3 * y_bar * huMoments[5] + 2 * y_bar * y_bar * m01;

        float n20 = mu20 / (m00 * m00);
        float n02 = mu02 / (m00 * m00);
        float n11 = mu11 / (m00 * m00);
        float n30 = mu30 / (m00 * m00 * m00);
        float n12 = mu12 / (m00 * m00 * m00);
        float n21 = mu21 / (m00 * m00 * m00);
        float n03 = mu03 / (m00 * m00 * m00);

        float h1 = n20 + n02;
        float h2 = (n20 - n02) * (n20 - n02) + 4 * n11 * n11;
        float h3 = (n30 - 3 * n12) * (n30 - 3 * n12) + (3 * n21 - n03) * (3 * n21 - n03);
        float h4 = (n30 + n12) * (n30 + n12) + (n21 + n03) * (n21 + n03);
        float h5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) * (n30 + n12) - 3 * (n21 + n03) * (n21 + n03)) + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03));
        float h6 = (n20 - n02) * ((n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03)) + 4 * n11 * (n30 + n12) * (n21 + n03);
        float h7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) * (n30 + n12) - 3 * (n21 + n03) * (n21 + n03)) - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03));

        // Normalize Hu moments
        float maxHu = fmaxf(fmaxf(fmaxf(h1, h2), fmaxf(h3, h4)), fmaxf(fmaxf(h5, h6), h7));
        if (maxHu != 0) {
            huMoments[0] = h1 / maxHu;
            huMoments[1] = h2 / maxHu;
            huMoments[2] = h3 / maxHu;
            huMoments[3] = h4 / maxHu;
            huMoments[4] = h5 / maxHu;
            huMoments[5] = h6 / maxHu;
            huMoments[6] = h7 / maxHu;
        }
    }
}