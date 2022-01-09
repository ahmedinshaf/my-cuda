//nessary files
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//prinf function
#include <stdio.h>


//device code
__global__ void hello_cuda()
{
    printf("Hello cuda world \n");
}

//host code
int main()
{
    //nx : number of threds in "x" dimension
    //ny : number of threds in "y" direction
    int nx,ny;

    nx = 16;
    ny = 4 ;    

    /*
    * Limitations 
    FOR BLOCK
    x,y <= 1024
    z <= 64
    max number for block =1024 <= ( x*y*z )

    FOR DIMENSION
    x <= 2^32 - 1
    y <= 65536
    z <= 65536 

    * IF NOT ADHIRE TO LIMITATIONS
    ** kernal lanuch failur
    */


   //threadIdx
   // A | B



    // calculating 
    dim3 block(8,2);
    dim3 grid(nx/block.x,ny/block.y);

    //many threds in block
    //dim3 block(4);

    //many blocks in grid
    //dim3 grid(8); 



    //kernal launch parameter
    hello_cuda <<<grid,block>>> ();

    //wait until (sync)
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
