//nessary files
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//prinf function
#include <stdio.h>


// not change amoung one block
__global__ void blockIdxCuda()
{
    printf("blockIdx.x : %d , blockIdx.y : %d , blockIdx.z : %d \n blockDim.x : %d , blockDim.y : %d \n  gridDim.x : %d , gridDim.y : %d " ,
     blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,gridDim.x,gridDim.y);
}

int main()
{
    int nx,ny;
    nx = 16 ;
    ny = 16 ;

    dim3 block(8,8);
    dim3 grid(nx/block.x,ny/block.y);

    //launch kernal

    blockIdxCuda<<<block,grid>>> ();
    
    cudaDeviceSynchronize();
    cudaDeviceReset();


}
  

//blockDIm
//gridDim are same  for all threds