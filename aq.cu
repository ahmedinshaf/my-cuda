#include <stdio.h>
#include <cuda.h>

_global_ void vecAdd( float *A, float *B,float *C) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        C[i] = A[i] + B[i];
}

int main(int argc,char** argv){

        int N=atoi(argv[1]); //no. of total elements
        int thread= atoi(argv[2]);
        float *a, *b, *c;
        float *dev_a, *dev_b, *dev_c;

        a = (float*)malloc(sizeof(float) * N);
        b = (float*)malloc(sizeof(float) * N);
        c = (float*)malloc(sizeof(float) * N);

        for(int i = 0 ; i < N; i++){
                a[i] = i*1.0+1;
                b[i] = i*1.0+1;
        }

        cudaMalloc((void**)&dev_a, sizeof(float) * N);
        cudaMalloc((void**)&dev_b, sizeof(float) * N);
        cudaMalloc((void**)&dev_c, sizeof(float) * N);

        cudaMemcpy(dev_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

        int block = (N + thread-1) / thread;
        vecAdd<<<block,thread>>>(dev_a,dev_b,dev_c);

        cudaMemcpy(c, dev_c, sizeof(float) * N, cudaMemcpyDeviceToHost);

        printf("Thread Size = %d\n",thread);
        printf("Block Size = %d\n",block);

        //for(int i = 0; i <N ; i++){
        //      printf("%f + %f = %f\n",a[i],b[i],c[i]);
        //}
        for(int i =0 ; i <5 ; i++){
                printf("%f + %f = %f\n",a[i],b[i],c[i]);
  }

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

        return 0;
}