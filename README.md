# CNN_CUDA

Code Compilation:

nvcc -gencode arch=compute_35,code=\"sm_35,compute_35\" cuda_functions.cu -o cnn
