
#include <stdio.h>
#include <cuda.h>
__global__ void sum_array_gpu( int * a, int * b,int *c, int size)
{
    int gid = blockIdx.x * blockDim.x +threadIdx .x;

    c[gid] = a[gid] + b[gid];

}

void sum_array_cpu (int *a , int *b , int *c, int size)
{
    for(int i = 0 ; i< size ; i++)
    {
        c[i] = a[i] + b[i];
    }
}

 bool compare_arrays(int * a, int *b,int size)
{
    for(int i =0 ;i< size ;i++)
    {
        if(a[i] != b[i])
        {
           // printf("Error in Sum %d , %d", a[i],b[i]);
            return false;
        }
    }
    //printf("Arrays are same");
    return true;
}

int main()
{
    //total size
    long size = 1024;
    int block_size = 1;
    bool isError= false; 

    while (!isError)
    {
    int NO_BYTES = size * sizeof(int);

    //host pointers
    int * h_a , *h_b , * gpu_result , *h_c;

    h_a = (int*) malloc(NO_BYTES);
    h_b = (int*) malloc(NO_BYTES);
    gpu_result = (int*) malloc(NO_BYTES);
    h_c = (int*) malloc(NO_BYTES);

    time_t t;
    srand((unsigned)time(&t));
    for(int i=0;i<size;i++)
    {
        h_a[i]  = (int (rand() & 0xFF));
    }

    for(int i=0;i<size;i++)
    {
        h_b[i]  = (int (rand() & 0xFF));
    }

    sum_array_cpu(h_a,h_b,h_c,size);

    memset (gpu_result,0,NO_BYTES);

    int *d_a, *d_b, *d_c;
    cudaMalloc((int **)&d_a,NO_BYTES);
    cudaMalloc((int **)&d_b,NO_BYTES);
    cudaMalloc((int **)&d_c,NO_BYTES);

    cudaMemcpy(d_a,h_a,NO_BYTES,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,NO_BYTES,cudaMemcpyHostToDevice);

    dim3 block (block_size);
    dim3 grid ((size/block.x)+1);

    //calling GPU
    sum_array_gpu<<<1,block>>>(d_a,d_b,d_c,size);
    cudaDeviceSynchronize();

    cudaMemcpy(gpu_result,d_c,NO_BYTES,cudaMemcpyDeviceToHost);
    
    //copmare
    isError = compare_arrays(gpu_result,h_c,size);
    if(isError){
        printf("Limit in %d ", block_size);
    }

    block_size++;

    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    free(gpu_result);
    }
   printf("Exection finshed");
}