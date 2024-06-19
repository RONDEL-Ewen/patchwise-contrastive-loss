/**
 * LossComputation.cu
 * Implements CUDA kernels and functions for computing cosine similarity between embeddings and calculating the contrastive loss.
 */

#include "NaiveParallelization.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

// ==================== Cosine Similarity ====================

__device__ float cosineSimilarityNaivePara(float *vecA, float *vecB, int length) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (int i = 0; i < length; ++i) {
        dot += vecA[i] * vecB[i];
        denom_a += vecA[i] * vecA[i];
        denom_b += vecB[i] * vecB[i];
    }
    return dot / (sqrtf(denom_a) * sqrtf(denom_b));
}

__global__ void computeSimilarityNaivePara(float *embeddings, float *similarities, int numEmbeddings, int embLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int refIdx = 0;  // Assume that the reference embedding is always at index 0
    if (idx < numEmbeddings && idx > 0) { // Start from 1 to skip the reference embedding
        similarities[idx] = cosineSimilarityNaivePara(&embeddings[refIdx * embLength], &embeddings[idx * embLength], embLength);
    }
}

// ==================== Contrastive Loss ====================

__global__ void computeContrastiveLossNaivePara(float *similarities, float *loss, int numNegatives, float margin) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float posSim = similarities[1];  // Index 1 is the positive embedding similarity
    float negSimSum = 0.0;

    // Compute sum of all negative similarities
    for (int i = 2; i <= numNegatives + 1; i++) {
        negSimSum += exp(similarities[i]);
    }

    // Only the first thread performs the final loss computation
    if (tid == 0) {
        *loss = -log(exp(posSim) / (exp(posSim) + negSimSum));
    }
}

// ==================== computeLoss ====================

void computeLossNaivePara(float *embeddings, float &loss, int numEmbeddings, int embLength, float margin) {
    
    float *d_embeddings, *d_similarities, *d_loss;
    cudaMalloc(&d_embeddings, numEmbeddings * embLength * sizeof(float));
    cudaMalloc(&d_similarities, numEmbeddings * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

    cudaMemcpy(d_embeddings, embeddings, numEmbeddings * embLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_loss, 0, sizeof(float));  // Initialize loss to zero

    dim3 blockSize(256);
    dim3 gridSize((numEmbeddings + blockSize.x - 1) / blockSize.x);
    computeSimilarityNaivePara<<<gridSize, blockSize>>>(d_embeddings, d_similarities, numEmbeddings, embLength);

    cudaDeviceSynchronize();  // Ensure similarity calculation is done

    computeContrastiveLossNaivePara<<<1, 1, (numEmbeddings - 1) * sizeof(float)>>>(d_similarities, d_loss, numEmbeddings - 2, margin);

    cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_embeddings);
    cudaFree(d_similarities);
    cudaFree(d_loss);
}