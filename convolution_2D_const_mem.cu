# include <stdio.h>

#define FILTER_RADIUS 1
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1]; //= {0,1,0,1,0,1,0,1};

__global__
void convolution_2D_basic_kernel(float *N, float *P, int r, int width, int height){
  int outCol = blockDim.x * blockIdx.x + threadIdx.x;
  int outRow = blockDim.y * blockIdx.y + threadIdx.y;
  float output = 0.0;

  if (outCol < width && outRow < height){
    for (int fRow=0; fRow <2*r+1; fRow++){
      for (int fCol=0; fCol <2*r+1; fCol++){
        int inRow = outRow -r + fRow;
        int inCol = outCol -r + fCol;
        if (inRow>=0 && inRow < height && inCol>=0 && inCol < width){
          output += N[inRow*width + inCol] * F[fRow][fCol];//[fRow*(2*r+1) + fCol];
            }
      }
    }
    P[outRow*width + outCol] = output;

  }
}

int main(){
  float *N_h, *P_h;
  int width, height;
  int r = FILTER_RADIUS;
  float F_h[2*r+1][2*r+1];

  width = 8;
  height = 8;

  N_h = (float *)malloc(sizeof(float) * width * height);
  P_h = (float *)malloc(sizeof(float) * width * height);

  int count = 0;

  for (int i = 0; i < width*height; i++){
    N_h[i] = count++;
  }

  for (int i=0; i< (2*FILTER_RADIUS +1); i++){
  for (int j=0; j< (2*FILTER_RADIUS +1); j++){
    if ((i+j) % 2 == 0){ F_h[i][j] = 0;}
    else { F_h[i][j] = 1;}
  }
  }

  printf("\nInput: \n");
  for (int i = 0; i < width*height; i++){
    printf("%f \t", N_h[i]);
  }
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

  dim3 threadsPerBlock(2,2);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  convolution_2D_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(N,P,r,width,height);

  cudaMemcpy(P_h, P,sizeof(float) * width * height, cudaMemcpyDeviceToHost);

  printf("\nOutput:\n");
  for (int i = 0; i < width * height; i++) {
        printf("%f\n", P_h[i]);
    }

    cudaFree(P);
    cudaFree(N);
    free(N_h);
    free(P_h);


  return 0;
}
