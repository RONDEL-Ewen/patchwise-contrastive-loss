# Makefile for the Patchwise Contrastive Loss project

# Compiler
NVCC = nvcc

# Paths
INCLUDES = -IUtils -ICPU -IGPU_Naive -IGPU_Shared_Memory

# Source files
CPU_SOURCES = CPU/PatchesExtraction.cu CPU/FeaturesExtraction.cu CPU/LossComputation.cu
GPU_NAIVE_SOURCES = GPU_Naive/PatchesExtraction.cu GPU_Naive/FeaturesExtraction.cu GPU_Naive/LossComputation.cu
GPU_SHARED_SOURCES = GPU_Shared_Memory/FeaturesExtraction.cu
MAIN_SOURCE = Main.cu

# Executable
EXECUTABLE = main.exe

# Default parameters
MODE ?= 0
IN_IMG ?= "Images/Inputs/input.png"
OUT_IMG ?= "Images/Outputs/output.png"
NUM_NEG ?= 5
PATCH_SIZE ?= 128

# Commands
$(EXECUTABLE): $(MAIN_SOURCE) $(CPU_SOURCES) $(GPU_NAIVE_SOURCES) $(GPU_SHARED_SOURCES)
	$(NVCC) -o $@ $(MAIN_SOURCE) $(INCLUDES) $(CPU_SOURCES) $(GPU_NAIVE_SOURCES) $(GPU_SHARED_SOURCES)

# Execute program
run:
	./$(EXECUTABLE) --mode $(MODE) --inImg $(IN_IMG) --outImg $(OUT_IMG) --numNeg $(NUM_NEG) --patchSize $(PATCH_SIZE)

# Clean compiled files
clean:
	del /f $(EXECUTABLE)
