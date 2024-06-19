# patchwise-contrastive-loss
Implementation of the Patchwise Contrastive Loss in CUDA (project for the GPU Programming course @PoliTo). <br><br>

## Patchwise Contrastive Loss

### How does it work ?
<br>

**1st Step:** Patches Extraction <br>

The algorithm extracts 3 different types of patches: <br>
- 1 reference patch, on the input image; <br>
- 1 positive patch, on the output image (with the same coordinates as the reference patch); <br>
- N random negative patches, on the output image. <br> <br>

**2nd Step:** Features Extraction <br>

For each patch, the algorithm extract 5 different features: <br>
- the mean RGB values; <br>
- a color histogram; <br>
- an oriented gradients histogram; <br>
- a texture histogram; <br>
- the 7 Hu moments. <br>
Then, it concatenates everything to create one embedding per patch. <br> <br>

**3rd Step:** Loss Computation <br>

The algorithm calculates the similarity for each pair of patches possible using the reference patch (with the positive patch and with every negative patch). And finally, it computes the contrastive loss. <br><br>

### Implementations
<br>

**Version 1:** Single-Threaded <br>
This CPU based version is executed sequentially. <br><br>

**Version 2:** Naive Parallelization <br>
This GPU based version takes advantage of the multi-streams achitecture. <br><br>

**Version 3:** Shared Memory <br>
This GPU based version takes advantage of the multi-streams architecture and uses the shared memory. <br><br>

## How to run the code ?

### Method 1 - Makefile
<br>

A Makefile has been added to the project to facilitate the compilation and execution of the code.

Here is the list of commands: <br><br>
To compile the code:
```
make
```
To run the code (with default settings):
```
make run
```
To clean the compiled files:
```
make clean
```

Some optional parameters are also available with the `make run` command: <br>
`MODE` : The mode in which to run the code (0: single-thread, 1: naive parallelization, 2: shared memory). <br>
`IN_IMG` : The path of the input image (relative to the root of the project). <br>
`OUT_IMG` : The path of the output image (relative to the root of the project). <br>
`NUM_NEG` : The number of negative patches. <br>
`PATCH_SIZE` : The size (width & height) of the patches. <br>

Here is an example of a command using some parameters:
```
make run MODE=1 NUM_NEG=50 PATCH_SIZE=128
```

### Method 2 - nvcc
<br>

To compile the code:
```
nvcc -o main ^
     Main.cu ^
     -IUtils ^
     -ICPU ^
     CPU/PatchesExtraction.cu ^
     CPU/FeaturesExtraction.cu ^
     CPU/LossComputation.cu ^
     -IGPU_Naive ^
     GPU_Naive/PatchesExtraction.cu ^
     GPU_Naive/FeaturesExtraction.cu ^
     GPU_Naive/LossComputation.cu ^
     -IGPU_Shared_Memory ^
     GPU_Shared_Memory/FeaturesExtraction.cu
```
*This command works on Windows. On Linux, replace all `^` by `\`.* <br>

To run the code (with default settings):
```
main.exe
```

Some optional parameters are also available: <br>
`--mode` : The mode in which to run the code (0: single-thread, 1: naive parallelization, 2: shared memory). <br>
`--inImg` : The path of the input image (relative to the root of the project). <br>
`--outImg` : The path of the output image (relative to the root of the project). <br>
`--numNeg` : The number of negative patches. <br>
`--patchSize` : The size (width & height) of the patches. <br>

Here is an example of a command using some parameters:
```
main.exe --mode 1 --numNeg 50 --patchSize 128
```
