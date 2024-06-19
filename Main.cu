/**
 * Main.cu
 * Entrypoint of the Patchwise Contrastive Loss project using CUDA.
 * This file contains the main function that orchestrates the flow of the program,
 * coordinating the extraction of patches, feature computation, and loss calculation.
 */

#include "CPU/BaseImplementation.h"
#include "GPU_Naive/NaiveParallelization.h"
#include "GPU_Shared_Memory/SharedMemory.h"

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <float.h>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "Utils/stb_image.h"

// ==================== Main ====================

int main(int argc, char *argv[]) {

    // Default variables
    // Mode
    // 0 = CPU Sequential
    // 1 = GPU Naive Parallelization
    // 2 = GPU Shared Memory
    int mode = 0;
    // Paths for input & output images
    const char* inputImagePath = "Images/Inputs/input.png";
    const char* outputImagePath = "Images/Outputs/output.png";
    // Number of negative patches
    int numNegativePatches = 5;
    // Patches size (width = height)
    int patchSize = 128;

    // Custom variables
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
            mode = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--inImg" && i + 1 < argc) {
            inputImagePath = argv[++i];
        } else if (std::string(argv[i]) == "--outImg" && i + 1 < argc) {
            outputImagePath = argv[++i];
        } else if (std::string(argv[i]) == "--numNeg" && i + 1 < argc) {
            numNegativePatches = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--patchSize" && i + 1 < argc) {
            patchSize = std::atoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0] << " [--mode number] [--inputImagePath path] [--outputImagePath path] [--numNegativePatches number] [--patchSize size]" << std::endl;
            return 1;
        }
    }

    // Load images using stb_image
    int width, height, channels;
    unsigned char *inputImage = stbi_load(inputImagePath, &width, &height, &channels, 0);
    unsigned char *outputImage = stbi_load(outputImagePath, &width, &height, &channels, 0);
    if (!inputImage || !outputImage) {
        printf("Error loading images\n");
        stbi_image_free(inputImage);
        stbi_image_free(outputImage);
        return -1;
    }

    // Assuming images are loaded with 3 channels (RGB)
    int numChannels = 3;

    // Allocate memory for patches
    unsigned char *refPatch, *posPatch, *negPatches;
    cudaMallocManaged(&refPatch, patchSize * patchSize * numChannels);
    cudaMallocManaged(&posPatch, patchSize * patchSize * numChannels);
    cudaMallocManaged(&negPatches, numNegativePatches * patchSize * patchSize * numChannels);

    // Allocate memory for embeddings
    float *embeddings;
    cudaMallocManaged(&embeddings, (2 + numNegativePatches) * 1070 * sizeof(float));

    // Allocate memory for loss
    float loss;

    if (mode == 0) {

        // CPU Sequential
        extractPatches(inputImage, outputImage, width, height, patchSize, numChannels, numNegativePatches, refPatch, posPatch, negPatches);
        extractFeatures(refPatch, posPatch, negPatches, embeddings, numNegativePatches+2, patchSize, patchSize, 3);
        computeLoss(embeddings, loss, numNegativePatches+2, 1070, 1.0);

    } else if (mode == 1) {

        // GPU Naive Parallelization
        extractPatchesNaivePara(inputImage, outputImage, width, height, patchSize, numChannels, numNegativePatches, refPatch, posPatch, negPatches);
        extractFeaturesNaivePara(refPatch, posPatch, negPatches, embeddings, numNegativePatches+2, patchSize, patchSize, 3);
        computeLossNaivePara(embeddings, loss, numNegativePatches+2, 1070, 1.0);

    } else if (mode == 2) {

        // GPU Shared Memory
        extractPatchesNaivePara(inputImage, outputImage, width, height, patchSize, numChannels, numNegativePatches, refPatch, posPatch, negPatches);
        extractFeaturesShared(refPatch, posPatch, negPatches, embeddings, numNegativePatches+2, patchSize, patchSize, 3);
        computeLossNaivePara(embeddings, loss, numNegativePatches+2, 1070, 1.0);

    } else {
        printf("Invalid mode");
        return 1;
    }

    // Display loss
    printf("\nPatcwise Contrastive Loss: %.5f\n", loss);

    // Free device memory
    cudaFree(refPatch);
    cudaFree(posPatch);
    cudaFree(negPatches);
    cudaFree(embeddings);
    stbi_image_free(inputImage);
    stbi_image_free(outputImage);

    return 0;
}