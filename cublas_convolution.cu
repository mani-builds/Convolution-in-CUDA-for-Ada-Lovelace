#include <cstdlib>
#include <stdio.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"

#define CHECK(call) \
do {                \
  const cudaError_t error_code = call; \
  if (error_code != cudaSuccess) {     \
    printf("CUDA Error:\n");            \
    printf("    File:       %s\n", __FILE__);   \
    printf("    Line:       %d\n", __LINE__);   \
    printf("    Error code: %d\n", error_code); \
    printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
    exit(1);\
  }\
} while(0)

#define IDX2C(i, j, ld) ((j)*(ld) + (i))


// void im2col() {
//   // Transform the input to matrix where each col is flattened patch

// }
__global__ void convolution_kernel() {


}

int main() {


  return EXIT_SUCCESS;

}





