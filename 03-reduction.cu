#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

__global__ void reduce_0x400(float* rr)
{
	volatile uint32_t tid = threadIdx.x;
	volatile uint32_t bid = blockIdx.x;
	volatile uint32_t i = 2 * blockDim.x * bid + tid;

	__shared__ float ss[0x400];

	ss[tid] = rr[i] + rr[i + 0x400];
	__syncthreads();

	if(tid & 0x200)
		return;
	ss[tid] += ss[tid + 0x200];
	__syncthreads();
	if(tid & 0x100)
		return;
	ss[tid] += ss[tid + 0x100];
	__syncthreads();
	if(tid & 0x080)
		return;
	ss[tid] += ss[tid + 0x080];
	__syncthreads();
	if(tid & 0x040)
		return;
	ss[tid] += ss[tid + 0x040];
	__syncthreads();
	if(tid & 0x020)
		return;
	ss[tid] += ss[tid + 0x020];
	ss[tid] += ss[tid + 0x010];
	ss[tid] += ss[tid + 0x008];
	ss[tid] += ss[tid + 0x004];
	ss[tid] += ss[tid + 0x002];
	ss[tid] += ss[tid + 0x001];
	if(0 == tid)
		rr[bid] = ss[0];
}

static void init_elements(float* rr, uint32_t numElements)
{
	srand48(time(0));
	while(numElements --)
	{
		*rr++ = drand48();
	}
}

static float host_sum_elements(float* rr, uint32_t numElements)
{
	float res = 0;
	while(numElements --)
	{
		res += *rr++;
	}
	return res;
}

static int dev_alloc(void** buff, size_t num_bytes)
{
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void**)buff, num_bytes);
	if(cudaSuccess != err) {
		fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
		return 1;
	}
	return 0;
}

static int copy_input_to_device(void* dest, void* src, size_t numBytes)
{
	cudaError_t err = cudaMemcpy(dest, src, numBytes, cudaMemcpyHostToDevice);
	if(cudaSuccess != err)
	{
		fprintf(stderr, "copy input - cudaMemcpy: %s\n", cudaGetErrorString(err));
		return 3;
	}
	return 0;
}

static int copy_output_to_host(void* dest, void* src, size_t numBytes)
{
	cudaError_t err = cudaMemcpy(dest, src, numBytes, cudaMemcpyDeviceToHost);
	if(cudaSuccess != err)
	{
		fprintf(stderr, "copy output - cudaMemcpy: %s\n", cudaGetErrorString(err));
		return 4;
	}
	return 0;
}

static void print_output(float* arr, size_t numElements)
{
	size_t i = 0;
	while(numElements --) {
		fprintf(stderr, "%04u: %g\n", i++, *arr++);
	}
}

main(void)
{
	cudaError_t err = cudaSuccess;
	uint32_t numElements = 4096;
	uint32_t data_size = numElements * sizeof(float);
	float rr[numElements];
	float* dev_rr = 0;
	int res = 0;
	float gold = 0;

	if((res = dev_alloc((void**)&dev_rr, data_size)))
		return res;

	init_elements(rr, numElements);
	gold = host_sum_elements(rr, 4096);

	do {
		if((res = copy_input_to_device(dev_rr, rr, data_size)))
			break;
    
		reduce_0x400<<<2, 1024>>>(dev_rr);
		if((res = copy_output_to_host(rr, dev_rr, data_size)))
			break;

		printf("GOLD: %.8g\n", gold);
		printf("CALC: %.8g\n", rr[0] + rr[1]);
	}while(0);

	err = cudaFree(dev_rr);

	if(cudaSuccess != err)
	{
		fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(err));
		return 2;
	}

	return res;
}


