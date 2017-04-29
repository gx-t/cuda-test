#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

struct ADD_ELEMENT {
	float a, b, c;
};

__global__ void vectorAdd(struct ADD_ELEMENT* el, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		el[i].c = el[i].a + el[i].b;
	}
}

static void init_elements(struct ADD_ELEMENT* arr, int numElements)
{
	while(numElements --)
	{
		arr->a = (float)rand()/RAND_MAX;
		arr->b = (float)rand()/RAND_MAX;
		arr->c = 0;
		arr ++;
	}

}

static int copy_input_to_device(void* dest, void* src, int numBytes)
{
	cudaError_t err = cudaMemcpy(dest, src, numBytes, cudaMemcpyHostToDevice);
	if(cudaSuccess != err)
	{
		fprintf(stderr, "copy input - cudaMemcpy: %s\n", cudaGetErrorString(err));
		return 3;
	}
	return 0;
}

static int copy_output_to_host(void* dest, void* src, int numBytes)
{
	cudaError_t err = cudaMemcpy(dest, src, numBytes, cudaMemcpyDeviceToHost);
	if(cudaSuccess != err)
	{
		fprintf(stderr, "copy output - cudaMemcpy: %s\n", cudaGetErrorString(err));
		return 4;
	}
	return 0;
}

main(void)
{
	cudaError_t err = cudaSuccess;
	int numElements = 50000;
	struct ADD_ELEMENT arr[numElements];
	struct ADD_ELEMENT* dev_arr = 0;
	int res = 0;

	init_elements(arr, numElements);

	err = cudaMalloc((void**)&dev_arr, sizeof(arr));
	if(cudaSuccess != err) {
		fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
		return 1;
	}

	do {
		if((res = copy_input_to_device(dev_arr, arr, sizeof(arr))))
			break;
    
	vectorAdd<<<numElements/200, 200>>>(dev_arr, numElements);

		if((res = copy_output_to_host(arr, dev_arr, sizeof(arr))))
			break;
	}while(0);

	err = cudaFree(dev_arr);

	if(cudaSuccess != err)
	{
		fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(err));
		return 2;
	}

	return res;
}


