#include <stdio.h>
#include <cuda.h>


// single block

// multiple block

__global__ void vectAdd(float* A, float* B, float* C){
        int i = threadIdx.x;
        int j = blockIdx.x * blockDim.x;
        int r = i + j;
        C[r] = A[r] + B[r];
}

int main(int argc, char** argv){
// atoi(argv[1])
// atoi(argv[2])

        int BLOCK_SIZE = 2;
        int THREAD_SIZE = 2;
        int TotalThreads = BLOCK_SIZE* THREAD_SIZE;
        float vectA[THREAD_SIZE], vectB[THREAD_SIZE], vectC[THREAD_SIZE];
        float *dev_a, *dev_b, *dev_c;

        cudaMalloc((void **)&dev_a, THREAD_SIZE * sizeof(float));
        cudaMalloc((void **)&dev_b, THREAD_SIZE * sizeof(float));
        cudaMalloc((void **)&dev_c, THREAD_SIZE * sizeof(float));

        for(int i = 0; i  < TotalThreads; i++){
                vectA[i] = 5.0 * i + 5.0;
                vectB[i] = 10.0 * i + 10.0;
        }

        cudaMemcpy(dev_a, vectA, THREAD_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, vectB, THREAD_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        vectAdd<<<BLOCK_SIZE, THREAD_SIZE>>>(dev_a, dev_b, dev_c);

        cudaMemcpy(vectC, dev_c, THREAD_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        for(int i = 0; i < TotalThreads; i++){
                printf("%0.1f VECTADD %0.1f = %0.1f\n",vectA[i], vectB[i], vectC[i]);
        }

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

        cudaDeviceReset();

        return 0;
}
