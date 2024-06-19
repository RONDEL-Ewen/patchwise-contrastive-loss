/**
 * PatchesFeatures.cu
 * Defines functions for extracting various patch features such as RGB averages,
 * color histograms, texture histograms, oriented gradient histograms, and Hu moments.
 * These features are extracted from the image patches and are used to form embeddings.
 */

#include "BaseImplementation.h"
#include <cstdio>
#include <math.h>

#define CH_BINS 256                     // Number of bins per channel (color histogram)
#define CHANNELS 3                      // Number of color channels (RGB) (color histogram)

#define GH_BINS 36                      // Number of bins for gradient orientations (oriented gradients histogram)
#define PI 3.14159265358979323846

#define TH_BINS 256                     // Number of bins for LBP values (texture histogram)

#define FEATURE_LENGTH (3 + CH_BINS * CHANNELS + GH_BINS + TH_BINS + 7)

// ==================== Mean RGB ====================

void calculateRGBMeans(unsigned char* patch, float* rgbMeans, int patchWidth, int patchHeight, int numChannels) {
    float rgb[3] = {0.0f, 0.0f, 0.0f};
    int totalPixels = patchWidth * patchHeight;

    for (int y = 0; y < patchHeight; y++) {
        for (int x = 0; x < patchWidth; x++) {
            int pixelIndex = (y * patchWidth + x) * numChannels;
            rgb[0] += patch[pixelIndex];
            rgb[1] += patch[pixelIndex + 1];
            rgb[2] += patch[pixelIndex + 2];
        }
    }

    rgbMeans[0] = (rgb[0] / totalPixels) / 255.0f; // Average Red normalized
    rgbMeans[1] = (rgb[1] / totalPixels) / 255.0f; // Average Green normalized
    rgbMeans[2] = (rgb[2] / totalPixels) / 255.0f; // Average Blue normalized
}

// ==================== Colors Histogram ====================

void calculateColorHistogram(unsigned char* patch, float* histogram, int patchWidth, int patchHeight, int numChannels) {
    int histogramSize = 256 * 3; // 256 bins for each RGB channel
    int totalPixels = patchWidth * patchHeight;

    for (int i = 0; i < histogramSize; i++) {
        histogram[i] = 0;
    }

    for (int y = 0; y < patchHeight; y++) {
        for (int x = 0; x < patchWidth; x++) {
            int pixelIndex = (y * patchWidth + x) * numChannels;
            histogram[patch[pixelIndex]]++;            // Histogram for Red
            histogram[256 + patch[pixelIndex + 1]]++;  // Histogram for Green
            histogram[512 + patch[pixelIndex + 2]]++;  // Histogram for Blue
        }
    }

    for (int i = 0; i < histogramSize; i++) {
        histogram[i] /= totalPixels; // Normalize the histogram
    }
}

// ==================== Oriented Gradients Histogram ====================

void calculateOrientedGradientHistogram(unsigned char* patch, float* histogram, int patchWidth, int patchHeight, int numChannels, int numBins) {
    float Gx, Gy;
    int totalPixels = (patchWidth - 2) * (patchHeight - 2); // Adjust for boundary conditions

    for (int i = 0; i < numBins; i++) {
        histogram[i] = 0;
    }

    for (int y = 1; y < patchHeight - 1; y++) {
        for (int x = 1; x < patchWidth - 1; x++) {
            int pixelIndex = (y * patchWidth + x) * numChannels;

            Gx = -1 * patch[pixelIndex - numChannels - patchWidth*numChannels]
                + 1 * patch[pixelIndex + numChannels - patchWidth*numChannels]
                - 2 * patch[pixelIndex - numChannels]
                + 2 * patch[pixelIndex + numChannels]
                - 1 * patch[pixelIndex - numChannels + patchWidth*numChannels]
                + 1 * patch[pixelIndex + numChannels + patchWidth*numChannels];

            Gy = -1 * patch[pixelIndex - patchWidth*numChannels - numChannels]
                - 2 * patch[pixelIndex - patchWidth*numChannels]
                - 1 * patch[pixelIndex - patchWidth*numChannels + numChannels]
                + 1 * patch[pixelIndex + patchWidth*numChannels - numChannels]
                + 2 * patch[pixelIndex + patchWidth*numChannels]
                + 1 * patch[pixelIndex + patchWidth*numChannels + numChannels];

            float angle = atan2f(Gy, Gx) * 180.0f / PI;
            if (angle < 0) angle += 360.0f;
            int bin = (int)(angle / (360.0f / numBins));
            histogram[bin]++;
        }
    }

    for (int i = 0; i < numBins; i++) {
        histogram[i] /= totalPixels; // Normalize the histogram
    }
}

// ==================== Texture Histogram ====================

unsigned char getLBPValue(unsigned char* patch, int x, int y, int patchWidth, int numChannels) {
    const int dx[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    const int dy[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
    unsigned char center = patch[(y * patchWidth + x) * numChannels];
    unsigned char code = 0;

    for (int i = 0; i < 8; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        unsigned char neighbor = patch[(ny * patchWidth + nx) * numChannels];
        code |= (neighbor > center) << i;
    }

    return code;
}

void calculateTextureHistogram(unsigned char* patch, float* histogram, int patchWidth, int patchHeight, int numChannels, int numBins) {
    int totalPixels = (patchWidth - 2) * (patchHeight - 2);  // Adjust for boundary conditions

    for (int i = 0; i < numBins; i++) {
        histogram[i] = 0;
    }

    for (int y = 1; y < patchHeight - 1; y++) {
        for (int x = 1; x < patchWidth - 1; x++) {
            unsigned char lbpValue = getLBPValue(patch, x, y, patchWidth, numChannels);
            histogram[lbpValue]++;
        }
    }

    for (int i = 0; i < numBins; i++) {
        histogram[i] /= totalPixels; // Normalize the histogram
    }
}

// ==================== Hu Moments ====================

unsigned char rgbToGray(unsigned char r, unsigned char g, unsigned char b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;  // Standards coefficients for RGB to Grayscale conversion
}

void calculateHuMoments(unsigned char* patch, float* huMoments, int patchWidth, int patchHeight, int numChannels) {
    float m00 = 0, m10 = 0, m01 = 0, m11 = 0, m20 = 0, m02 = 0, m30 = 0, m12 = 0, m21 = 0, m03 = 0;

    for (int y = 0; y < patchHeight; y++) {
        for (int x = 0; x < patchWidth; x++) {
            int idx = y * patchWidth + x;
            unsigned char grayValue;
            if (numChannels == 1) {
                grayValue = patch[idx * numChannels];
            } else if (numChannels == 3) {
                grayValue = rgbToGray(patch[idx * numChannels], patch[idx * numChannels + 1], patch[idx * numChannels + 2]);
            }

            m00 += grayValue;
            m10 += grayValue * x;
            m01 += grayValue * y;
            m11 += grayValue * x * y;
            m20 += grayValue * x * x;
            m02 += grayValue * y * y;
            m30 += grayValue * x * x * x;
            m12 += grayValue * x * y * y;
            m21 += grayValue * x * x * y;
            m03 += grayValue * y * y * y;
        }
    }

    // Calculate centroid
    float x_bar = m10 / m00;
    float y_bar = m01 / m00;

    // Central moments
    float mu11 = m11 - y_bar * m10;
    float mu20 = m20 - x_bar * m10;
    float mu02 = m02 - y_bar * m01;
    float mu30 = m30 - 3 * x_bar * m20 + 2 * x_bar * x_bar * m10;
    float mu12 = m12 - x_bar * m02 - 2 * y_bar * m11 + 2 * y_bar * y_bar * m10;
    float mu21 = m21 - y_bar * m20 - 2 * x_bar * m11 + 2 * x_bar * x_bar * m01;
    float mu03 = m03 - 3 * y_bar * m02 + 2 * y_bar * y_bar * m01;

    // Normalized central moments
    float n11 = mu11 / pow(m00, 1.5);
    float n20 = mu20 / pow(m00, 2.0);
    float n02 = mu02 / pow(m00, 2.0);
    float n30 = mu30 / pow(m00, 2.5);
    float n12 = mu12 / pow(m00, 2.5);
    float n21 = mu21 / pow(m00, 2.5);
    float n03 = mu03 / pow(m00, 2.5);

    // Hu moments calculation
    huMoments[0] = n20 + n02;
    huMoments[1] = pow(n20 - n02, 2) + 4 * pow(n11, 2);
    huMoments[2] = pow(n30 - 3*n12, 2) + pow(3*n21 - n03, 2);
    huMoments[3] = pow(n30 + n12, 2) + pow(n21 + n03, 2);
    huMoments[4] = (n30 - 3*n12) * (n30 + n12) * (pow(n30 + n12, 2) - 3*pow(n21 + n03, 2)) + (3*n21 - n03) * (n21 + n03) * (3*pow(n30 + n12, 2) - pow(n21 + n03, 2));
    huMoments[5] = (n20 - n02) * ((n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03)) + 4*n11 * (n30 + n12) * (n21 + n03);
    huMoments[6] = (3*n21 - n03) * (n30 + n12) * (pow(n30 + n12, 2) - 3*pow(n21 + n03, 2)) - (n30 - 3*n12) * (n21 + n03) * (3*pow(n30 + n12, 2) - pow(n21 + n03, 2));

    // Normalize Hu moments to their maximum absolute value to avoid scale variations
    float maxHu = 0;
    for (int i = 0; i < 7; i++) {
        if (fabs(huMoments[i]) > maxHu) maxHu = fabs(huMoments[i]);
    }

    if (maxHu != 0) {
        for (int i = 0; i < 7; i++) {
            huMoments[i] /= maxHu;
        }
    }
}

// ==================== extractFeatures ====================

void extractFeatures(unsigned char* refPatch, unsigned char* posPatch, unsigned char* negPatches, float* embeddings, int numPatches, int patchWidth, int patchHeight, int numChannels) {
    // Extract features for each patch
    for (int p = 0; p < numPatches; p++) {
        unsigned char* currentPatch = (p == 0) ? refPatch : (p == 1) ? posPatch : negPatches + (p - 2) * patchWidth * patchHeight * numChannels;

        // Extract normalized mean RGB values from current patch
        float rgbMeans[CHANNELS] = {0};
        calculateRGBMeans(currentPatch, rgbMeans, patchWidth, patchHeight, numChannels);

        // Extract normalized color histogram from current patch
        float colorHistogram[CH_BINS * CHANNELS] = {0};
        calculateColorHistogram(currentPatch, colorHistogram, patchWidth, patchHeight, numChannels);

        // Extract normalized oriented gradients histogram from current patch
        float orientedGradientHistogram[GH_BINS] = {0};
        calculateOrientedGradientHistogram(currentPatch, orientedGradientHistogram, patchWidth, patchHeight, numChannels, GH_BINS);

        // Extract normalized texture histogram from current patch
        float textureHistogram[TH_BINS] = {0};
        calculateTextureHistogram(currentPatch, textureHistogram, patchWidth, patchHeight, numChannels, TH_BINS);

        // Extract normalized Hu moments from current patch
        float huMoments[7] = {0};
        calculateHuMoments(currentPatch, huMoments, patchWidth, patchHeight, numChannels);

        // Concatenate all features for the current patch
        int featureBaseIndex = p * FEATURE_LENGTH;
        for (int i = 0; i < CHANNELS; i++) embeddings[featureBaseIndex + i] = rgbMeans[i];
        for (int i = 0; i < CH_BINS*CHANNELS; i++) embeddings[featureBaseIndex + CHANNELS + i] = colorHistogram[i];
        for (int i = 0; i < GH_BINS; i++) embeddings[featureBaseIndex + CHANNELS + CH_BINS*CHANNELS + i] = orientedGradientHistogram[i];
        for (int i = 0; i < TH_BINS; i++) embeddings[featureBaseIndex + CHANNELS + CH_BINS*CHANNELS + GH_BINS + i] = textureHistogram[i];
        for (int i = 0; i < 7; i++) embeddings[featureBaseIndex + CHANNELS + CH_BINS*CHANNELS + GH_BINS + TH_BINS + i] = huMoments[i];
    }
}