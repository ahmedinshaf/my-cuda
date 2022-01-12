#include <iostream>
#include <ctime>
#include <random>
#include <stdio.h>


__global__ void dot_product_kernel(float *x, float *y, float *dot, unsigned int n)
{
	unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

	__shared__ float cache[256];

	double temp = 0.0;
	while(index < n){
		temp += x[index]*y[index];

		index += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}


	if(threadIdx.x == 0){
		atomicAdd(dot, cache[0]);
	}
}

int main()
{
	unsigned int n = 2;
	float *h_prod;
	float *d_prod;
	float *h_x, *h_y;
	float *d_x, *d_y;

	// allocate memory
	h_prod = (float*)malloc(sizeof(float)); 
	h_x = (float*)malloc(n*sizeof(float));
	h_y = (float*)malloc(n*sizeof(float));
	cudaMalloc((void**)&d_prod, sizeof(float));
	cudaMalloc((void**)&d_x, n*sizeof(float));
	cudaMalloc((void**)&d_y, n*sizeof(float));
	cudaMemset(d_prod, 0.0, sizeof(float));

	// fill host array with data
	for(unsigned int i=0;i<n;i++){
		h_x[i] = float(rand()%n) / n;
		h_y[i] = float(rand()%n) / n;
		// h_x[i] = 6;
		// h_y[i] = 3;
	}


	// timing variables
	float gpu_elapsed_time = 0.0;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);


	// copy data to device
	cudaMemcpy(d_x, h_x, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(gpu_start, 0);


	// call dot product kernel
	dim3 gridSize = 256;
	dim3 blockSize = 256;
	dot_product_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_prod, n);

	// copy data back to host
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);
	cudaMemcpy(h_prod, d_prod, sizeof(float), cudaMemcpyDeviceToHost);
	

	// report results
	std::cout<<"dot product computed on GPU is: "<<*h_prod<<" and took "<<gpu_elapsed_time<<std::endl;
	

	// run CPU based dot product to compare times to GPU code
	clock_t cpu_start = clock();
	double temp = 0.0;
	for(unsigned int i=0;i<n;i++){
		temp += h_x[i]*h_y[i];
	}
	//*h_prod = temp;
	clock_t cpu_stop = clock();
	clock_t cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;
	std::cout<<"dot product computed on CPU is: "<<temp<<" and took "<<cpu_elapsed_time<<std::endl;


	// free memory
	free(h_prod);
	free(h_x);
	free(h_y);
	cudaFree(d_prod);
	cudaFree(d_x);
	cudaFree(d_y);

}
