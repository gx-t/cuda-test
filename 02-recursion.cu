#include <stdio.h>
#include <stdint.h>

static __device__ uint32_t dev_fibonacci(uint32_t num)
{
	if(0 == num) {
		return 0;
	}
	if(1 == num) {
		return 1;
	}
	return dev_fibonacci(num - 1) + dev_fibonacci(num - 2);
}

__global__ void fibonacci(uint32_t* el, size_t numElements)
{
	int i = blockIdx.x;

	if(i < numElements)
	{
		el[i] = dev_fibonacci(i);
	}
}

static void init_elements(uint32_t* arr, uint32_t numElements)
{
	while(numElements --)
	{
		arr[numElements] = 0;
	}

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

static void print_output(uint32_t* arr, size_t numElements)
{
	size_t i = 0;
	while(numElements --) {
		fprintf(stderr, "%04u: %u\n", i++, *arr++);
	}
}

main(void)
{
	cudaError_t err = cudaSuccess;

	//VERY SLOW: real    0m12.028s for 32
	uint32_t numElements = 32;
	uint32_t arr[numElements];
	uint32_t* dev_arr = 0;
	int res = 0;

	err = cudaMalloc((void**)&dev_arr, sizeof(arr));
	if(cudaSuccess != err) {
		fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
		return 1;
	}

	init_elements(arr, numElements);

	do {
		if((res = copy_input_to_device(dev_arr, arr, sizeof(arr))))
			break;
    
		fibonacci<<<numElements, 1>>>(dev_arr, numElements);

		if((res = copy_output_to_host(arr, dev_arr, sizeof(arr))))
			break;

		print_output(arr, numElements);
	}while(0);

	err = cudaFree(dev_arr);

	if(cudaSuccess != err)
	{
		fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(err));
		return 2;
	}

	return res;
}


