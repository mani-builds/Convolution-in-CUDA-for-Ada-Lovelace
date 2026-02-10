# include <stdio.h>

#define TILE_DIM 4
#define FILTER_RADIUS 2
__constant__ float F[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];


__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height){
  int col =  blockIdx.x*TILE_DIM + threadIdx.x;
  int row =  blockIdx.y*TILE_DIM + threadIdx.y;
  // loading input tiles
  __shared__ float N_s[TILE_DIM][TILE_DIM];
  if (row < height && col <width){
  N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
}
  else{
    N_s[threadIdx.y][threadIdx.x] = 0.0f;
  }
__syncthreads();
// calculating output threads
// turning off the threads at the edge
  if (row < height && col <width){
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow<2*FILTER_RADIUS+1; fRow++){
      for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++){
        // covert the threadsIdx into 'int' to suppress the warning
        if ((int)threadIdx.x - FILTER_RADIUS + fCol >= 0 &&
            (int)threadIdx.x - FILTER_RADIUS + fCol < TILE_DIM &&
            (int)threadIdx.y - FILTER_RADIUS + fRow >= 0 &&
            (int)threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM ){
        Pvalue += F[fRow][fCol] * N_s[threadIdx.y - FILTER_RADIUS +fRow][threadIdx.x - FILTER_RADIUS +fCol];
      }
        else{
          if (row-FILTER_RADIUS + fRow >= 0 &&
              row-FILTER_RADIUS + fRow < height &&
              col-FILTER_RADIUS + fCol >= 0 &&
              col-FILTER_RADIUS + fCol < width){
            Pvalue += F[fRow][fCol] * N[(row-FILTER_RADIUS+fRow)*width + (col-FILTER_RADIUS + fCol)];
          }
        }
      }

    }
    P[row*width + col] = Pvalue;
  }
}

int main(){
  float *N_h, *P_h;
  int width, height;
  int r = FILTER_RADIUS;
  float F_h[2*r+1][2*r+1];

  width = 16;
  height = 16;

  N_h = (float *)malloc(sizeof(float) * width * height);
  P_h = (float *)malloc(sizeof(float) * width * height);

  // Create the 8x8 square of 1s in the middle (rows 4-11, cols 4-11)
    for (int r = 4; r < 12; r++) {
        for (int c = 4; c < 12; c++) {
            N_h[r * width + c] = 1.0f;
        }
    }

  // 5x5 Filter Matrix (Box Blur)
  for (int i=0; i< (2*FILTER_RADIUS +1); i++){
  for (int j=0; j< (2*FILTER_RADIUS +1); j++){
   F_h[i][j] = 1.0f / 25.0f;;
  }
  }

  // printf("\nInput: \n");
  // for (int i = 0; i < width*height; i++){
  //   printf("%f \t", N_h[i]);
  // }
  printf("\nFilter: \n");
  for (int i=0; i< (2*FILTER_RADIUS +1); i++){
  for (int j=0; j< (2*FILTER_RADIUS +1); j++){
    printf("%f \t", F_h[i][j]);
  }
  }

  float *N, *P;
  cudaMalloc(&N, sizeof(float) * width * height);
  cudaMalloc(&P, sizeof(float) * width * height);

  cudaMemcpy(N, N_h,sizeof(float) * width * height, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(F,F_h,sizeof(float) * (2*r+1) * (2*r+1));

  dim3 threadsPerBlock(TILE_DIM,TILE_DIM);
  dim3 blocksPerGrid((width + TILE_DIM - 1) / TILE_DIM,
                       (height + TILE_DIM - 1) / TILE_DIM);
  convolution_cached_tiled_2D_const_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(N,P,width,height);
  cudaMemcpy(P_h, P,sizeof(float) * width * height, cudaMemcpyDeviceToHost);

  printf("\nOutput:\n");
  for (int i = 0; i < width * height; i++) {
    if (i % width == 0){printf("\n"); }
    printf("%f\t", P_h[i]);
    }
    printf("\n");
    cudaFree(P);
    cudaFree(N);
    free(N_h);
    free(P_h);


  return 0;
}
