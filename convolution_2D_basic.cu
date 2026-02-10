# include <stdio.h>

__global__
void convolution_2D_basic_kernel(float *N, float *F, float *P, int r, int width, int height){
  int outCol = blockDim.x * blockIdx.x + threadIdx.x;
  int outRow = blockDim.y * blockIdx.y + threadIdx.y;
  float output = 0.0;
  if (outCol < width && outRow < height){
    for (int fRow=0; fRow <2*r+1; fRow++){
      for (int fCol=0; fCol <2*r+1; fCol++){
        int inRow = outRow -r + fRow;
        int inCol = outCol -r + fCol;
        if (inRow>=0 && inRow < height && inCol>=0 && inCol < width){
            output += N[inRow*width + inCol] * F[fRow*(2*r+1) + fCol];
            }
      }
    }

    P[outRow*width + outCol] = output;

  }
}

int main(){
  float *N_h, *F_h, *P_h;
  int r, width, height;
  width = 8;
  height = 8;
  r = 1;
  N_h = (float *)malloc(sizeof(float) * width * height);
  P_h = (float *)malloc(sizeof(float) * width * height);
  F_h = (float *)malloc(sizeof(float) * (2*r+1) * (2*r+1));

  int count = 0;

  for (int i = 0; i < width*height; i++){
    N_h[i] = count++;
  }

  for (int j=0; j< (2*r+1)*(2*r+1) ; j++){
    if (j % 2 == 0){ F_h[j] = 0;}
    else { F_h[j] = 1;}
  }

  printf("\nInput: \n");
  for (int i = 0; i < width*height; i++){
    printf("%f \t", N_h[i]);
  }
  printf("\nFilter: \n");
  for (int i = 0; i < (2*r+1)*(2*r+1); i++){
    printf("%f \t", F_h[i]);
  }

  float *N, *F, *P;
  cudaMalloc(&N, sizeof(float) * width * height);
  cudaMalloc(&P, sizeof(float) * width * height);
  cudaMalloc(&F, sizeof(float) * (2*r+1) * (2*r+1));

  cudaMemcpy(N, N_h,sizeof(float) * width * height, cudaMemcpyHostToDevice);
  cudaMemcpy(F, F_h,sizeof(float) * (2*r+1) * (2*r+1), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(2,2);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
  convolution_2D_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(N,F,P,r,width,height);

  cudaMemcpy(P_h, P,sizeof(float) * width * height, cudaMemcpyDeviceToHost);

  printf("\nOutput:\n");
  for (int i = 0; i < width * height; i++) {
        printf("%f\n", P_h[i]);
    }

    cudaFree(P);
    cudaFree(N);
    cudaFree(F);
    free(F_h);
    free(N_h);
    free(P_h);


  return 0;
}
