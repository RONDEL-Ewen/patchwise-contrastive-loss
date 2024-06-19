/**
 * LossComputation.cu
 * Implements functions for computing cosine similarity between embeddings and calculating the contrastive loss.
 */

#include "BaseImplementation.h"
#include <cstdio>
#include <math.h>

// ==================== Cosine Similarity ====================

float cosineSimilarity(float *vecA, float *vecB, int length) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (int i = 0; i < length; ++i) {
        dot += vecA[i] * vecB[i];
        denom_a += vecA[i] * vecA[i];
        denom_b += vecB[i] * vecB[i];
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

// ==================== Contrastive Loss ====================

void computeContrastiveLoss(float *similarities, float &loss, int numNegatives, float margin) {
    float posSim = similarities[1];  // Index 1 is the positive embedding similarity
    float negSimSum = 0.0;

    // Compute sum of all negative similarities
    for (int i = 2; i <= numNegatives + 1; i++) {
        negSimSum += exp(similarities[i]);
    }

    loss = -log(exp(posSim) / (exp(posSim) + negSimSum));
}

// ==================== computeLoss ====================

void computeLoss(float *embeddings, float &loss, int numEmbeddings, int embLength, float margin) {
    
    float *similarities = new float[numEmbeddings];
    // Calculate similarities
    for (int i = 1; i < numEmbeddings; i++) { // Start from 1 to skip the reference embedding
        similarities[i] = cosineSimilarity(embeddings, &embeddings[i * embLength], embLength);
    }

    computeContrastiveLoss(similarities, loss, numEmbeddings - 2, margin);

    delete[] similarities;
}