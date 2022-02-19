#include <stdio.h>

#define N 2048 * 2048 

__global__ void saxpy(int * a, int * b, int * c)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
        c[i] = 2 * a[i] + b[i];
}

__global__ void init(int * a, int * b, int * c)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride) {
        a[i] = 2;
        b[i] = 1;
        c[i] = 0;
    }
}

int main()
{
    int deviceId;
    cudaGetDevice(&deviceId);

    int threads_per_block = 128;
    int number_of_blocks = (N / threads_per_block) + 1;

    float *a, *b, *c;

    int size = N * sizeof (int);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);

    init<<<number_of_blocks, threads_per_block>>>(a, b, c);
    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );

    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
    
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
