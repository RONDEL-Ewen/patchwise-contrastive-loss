/**
 * PatchesExtraction.cu
 * Contains functions for extracting reference, positive, and negative patches from input and output images.
 */

#include "BaseImplementation.h"
#include <cstdlib>
#include <cstdio>
#include <ctime>

// ==================== Patches Coordinates ====================

void generateRandomCoordinates(int* coords, int width, int height, int patchSize, int numPatches) {
    srand(time(NULL));
    for (int i = 0; i < numPatches; i++) {
        coords[2 * i] = rand() % (width - patchSize);  // x coordinate
        coords[2 * i + 1] = rand() % (height - patchSize);  // y coordinate
    }
}

// ==================== Extract 1 Patch ====================

void extractPatch(const unsigned char* image, unsigned char* patch, int startX, int startY, int imageWidth, int patchSize, int numChannels) {
    for (int y = 0; y < patchSize; y++) {
        for (int x = 0; x < patchSize; x++) {
            int imageIndex = ((startY + y) * imageWidth + (startX + x)) * numChannels;
            int patchIndex = (y * patchSize + x) * numChannels;
            for (int channel = 0; channel < numChannels; channel++) {
                patch[patchIndex + channel] = image[imageIndex + channel];
            }
        }
    }
}

// ==================== extractPatches ====================

void extractPatches(const unsigned char* inputImage, const unsigned char* outputImage, int imageWidth, int imageHeight, int patchSize, int numChannels, int numNegativePatches, unsigned char* refPatch, unsigned char* posPatch, unsigned char* negPatches) {
    int numPatches = numNegativePatches + 2;
    int* hostCoords = (int*)malloc(numPatches * 2 * sizeof(int));

    // Generate coordinates
    generateRandomCoordinates(hostCoords, imageWidth, imageHeight, patchSize, numPatches);

    // Extract reference patch
    extractPatch(inputImage, refPatch, hostCoords[0], hostCoords[1], imageWidth, patchSize, numChannels);

    // Extract positive patch
    extractPatch(outputImage, posPatch, hostCoords[0], hostCoords[1], imageWidth, patchSize, numChannels);

    // Extract negative patches
    for (int i = 0; i < numNegativePatches; i++) {
        int offset = 2 * (i + 2);  // Offset for coordinates
        extractPatch(outputImage, negPatches + i * patchSize * patchSize * numChannels, 
                     hostCoords[offset], hostCoords[offset + 1], imageWidth, patchSize, numChannels);
    }

    free(hostCoords);
}