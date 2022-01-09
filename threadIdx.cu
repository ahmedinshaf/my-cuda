//nessary files
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//prinf function
#include <stdio.h>



//threadIdx values are calculated relative to grid
__global__ void threadIdxCuda()
{
    printf("threadIdx.x : %d , threadIdx.y : %d , threadIdx.z : %d  \n" , threadIdx.x,threadIdx.y,threadIdx.z);
}

int main()
{
    int nx,ny;
    nx = 16 ;
    ny = 16 ;

    dim3 block(8,8);
    dim3 grid(nx/block.x,ny/block.y);

    //launch kernal
    threadIdxCuda<<<block,grid>>> ();
    cudaDeviceSynchronize();
    cudaDeviceReset();


}