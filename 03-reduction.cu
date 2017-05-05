#include <stdio.h>
#include <stdint.h>

__global__ void reduce_0x20(float* rr)
{
	float __shared__ sdata[0x40];

	volatile uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t i   = blockDim.x * bid + tid;

	sdata[tid] = rr[i] + rr[i + 0x20];
	sdata[tid] = sdata[tid] + sdata[tid + 0x10];
	sdata[tid] = sdata[tid] + sdata[tid + 0x08];
	sdata[tid] = sdata[tid] + sdata[tid + 0x04];
	sdata[tid] = sdata[tid] + sdata[tid + 0x01];
	sdata[tid] = sdata[tid] + sdata[tid + 0x00];

	rr[bid]    = sdata[0x01];
}

static void init_elements(float* rr, uint32_t numElements)
{
	float i = 3.14;
	while(numElements --)
	{
		*rr++ = i;// ++;
	}
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

	if((res = dev_alloc((void**)&dev_rr, data_size)))
		return res;

	init_elements(rr, numElements);

	do {
		if((res = copy_input_to_device(dev_rr, rr, data_size)))
			break;
    
		reduce_0x20<<<64, 32>>>(dev_rr);
		reduce_0x20<<<1, 32>>>(dev_rr);

		if((res = copy_output_to_host(rr, dev_rr, data_size)))
			break;

		print_output(rr, 32);
	}while(0);

	err = cudaFree(dev_rr);

	if(cudaSuccess != err)
	{
		fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(err));
		return 2;
	}

	return res;
}


