/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Simplified by Cooper, to remove some of the obfuscating
 * code that only provides safety.
 */

/**
 * Vector addition: w = u + v.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of u and v into w. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *u, const float *v, float *w)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    w[i] = u[i] + v[i];
}

/**
 * Host main routine
 */
int
main(void)
{
    // Print the vector length to be used, and compute its size
    int numElements = 999999999;
    size_t size = numElements * sizeof(float);
	// Observe that this program is ever so slightly busted,
	// for reasons that will become apparent later.
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector u
    float *h_u = (float *)malloc(size);

    // Allocate the host input vector v
    float *h_v = (float *)malloc(size);

    // Allocate the host output vector w
    float *h_w = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_u[i] = rand()/(float)RAND_MAX;
        h_v[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector u
    float *d_u = NULL;
    cudaMalloc((void **)&d_u, size);

    // Allocate the device input vector v
    float *d_v = NULL;
    cudaMalloc((void **)&d_v, size);

    // Allocate the device output vector w
    float *d_w = NULL;
    cudaMalloc((void **)&d_w, size);

    // Copy the host input vectors u and v in host memory to the 
	// device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_u, d_v, d_w);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_w, d_w, size, cudaMemcpyDeviceToHost);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_u[i] + h_v[i] - h_w[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // Free device global memory
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);

    // Free host memory
    free(h_u);
    free(h_v);
    free(h_w);

    // Reset the device and exit
    cudaDeviceReset();

    printf("Done\n");
    return 0;
}
