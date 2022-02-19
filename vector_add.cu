#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

cudaDeviceProp getProps() {
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  return props;
}

int main()
{
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threadsPerBlock;
  size_t numberOfBlocks;
   
  threadsPerBlock = 1;
  numberOfBlocks = 1;
  cudaDeviceProp gpuProps = getProps();
  int threads_per_sm = N / gpuProps.multiProcessorCount;
  if (N % gpuProps.multiProcessorCount)
      threads_per_sm++;
      
  if (threads_per_sm > 1024) {
      int k = threads_per_sm / 1024;
      if (threads_per_sm % 1024)
          k++;
      numberOfBlocks = k * gpuProps.multiProcessorCount;
      threadsPerBlock = N / numberOfBlocks;
      if (N % numberOfBlocks)
          threadsPerBlock++;
  } else {
      threadsPerBlock = threads_per_sm;
      numberOfBlocks = gpuProps.multiProcessorCount;
  }
  
  if (threadsPerBlock % gpuProps.warpSize)
    threadsPerBlock = gpuProps.warpSize * (threadsPerBlock / gpuProps.warpSize + 1);

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  printf("Blocks %u Threads %u Multiprocs: %u\n", numberOfBlocks, threadsPerBlock, gpuProps.multiProcessorCount);
  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
