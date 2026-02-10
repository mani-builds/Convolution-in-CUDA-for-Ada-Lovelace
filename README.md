*   A convolutional layer from scratch in CUDA, optimize for Ada Lovelace architecture focusing on tensor cores 

Stage 1: Naive Implementation of a 2D/3D convolutional kernel in CUDA
It uses Global Memory
Profile: High Latency and Low Occupancy

Stage 2: Basic Optimizations 
Constant memory for the Kernel (since the data doesn't change during execution)
Shared Memory Tiling. Reduces global accessess dramatically 11x fewer
Profile these

Stage 3: im2col - convert a 2D image into an array by flattening it.

Stage 4: 
NVIDIA RTX 4060 (Ada Lovelace):
No. of Tensor cores:
How many per SM:
No. of SMs:

Tensor core GEMM using CUBLAS library
GEMM with WMMA: Use CUDA's WMMA API (Warp Matrix Multiply-Accumulate) for FP16 tensor ops.
Rules: Dimensions multiple of 8/16, FP16 inputs, enable CUDNN_TENSOR_OP_MATH if using cuDNN for comparison.
Layout: Use NHWC (channels last) for better tensor core perf.
Benchmark: Compare to cuDNN (cudnnConvolutionForward with TENSOR_OP_MATH). Aim for close-to-peak TFLOPS on Ampere.

Stage 5: make it an executable in linux like a library

Stage 6: Make it importable with pytorch
