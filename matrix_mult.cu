#include <stdio.h>

#define N  64

__global__ void matrixMulGPU( int * a, int * b, int * c )
{
  int x_offset = threadIdx.x + blockIdx.x * blockDim.x;
  int y_offset = threadIdx.y + blockIdx.y * blockDim.y;
  int x_step = blockDim.x * gridDim.x;
  int y_step = blockDim.y * gridDim.y;
  for (int x = x_offset; x < N; x += x_step) {
      for (int y = y_offset; y < N; y += y_step) {
          int sum = 0;
          for (int k = 0; k < N; k++) {
              sum += a[y * N  + k] * b[k * N + x];
          }
          c[y * N + x] = sum;
      }
  }
}

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu;
  int size = N * N * sizeof (int);

  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  size_t threads = 16;
  size_t blocks = N / threads;
  if (N % threads)
      blocks++;
  dim3 threads_per_block(threads, threads, 1);
  dim3 number_of_blocks(blocks, blocks, 1);

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

  cudaDeviceSynchronize();

  matrixMulCPU( a, b, c_cpu );

  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        printf("CPU=%d GPU=%d", c_cpu[row * N + col], c_gpu[row * N + col]);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}
