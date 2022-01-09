
#include <stdio.h>
#include <cuda.h>

// gpu function for sum array
__global__ void sum_array_gpu( int * a, int * b,int *c, int size)
{
    int gid = blockIdx.x * blockDim.x +threadIdx .x;

    c[gid] = a[gid] + b[gid];

}

//CPU function for sum array
void sum_array_cpu (int *a , int *b , int *c, int size)
{
    for(int i = 0 ; i< size ; i++)
    {
        c[i] = a[i] + b[i];
    }
}

//Comparing two arrays are same
/*
* return 
        true  => if arrays are different    
        false => if arrays are same
*/
 bool compare_arrays(int * a, int *b,int size)
{
    for(int i =0 ;i< size ;i++)
    {
        if(a[i] != b[i])
        {
            return true;
        }
    }
    return false;
}

int main()
{
    //total thread size 
    long size = 1;
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


    //assign random values for a and b array
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

    //calling GPU
    // single block , and block size increase from 1 - X ( have to determine X )    
    sum_array_gpu<<<1,block>>>(d_a,d_b,d_c,size);
    cudaDeviceSynchronize();

    cudaMemcpy(gpu_result,d_c,NO_BYTES,cudaMemcpyDeviceToHost);
    
    //compare
    isError = compare_arrays(gpu_result,h_c,size);
    if(isError){
        printf("Limit in block Size: %d ", block_size);
    }

    block_size++;
    size++;

    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    free(gpu_result);
    }
   printf("Exection finshed");
}