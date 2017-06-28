#include<iostream>
#include<bitset>

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

#include <cuda_runtime.h>
#include"device_functions.h"
#include"cuda.h"

#include"DeviceUtils.cuh"
#include"HostUtils.h"

__device__ unsigned long long compactKernel(bool * arr) {
	unsigned long long result = 0;
	unsigned long long temp;
	for (int i = 0; i < PIXELS; i++) {
		temp = arr[i];
		result |= temp << (PIXELS - i - 1);
	}
	return result;
}

__global__ void compactBatchKernel(bool * d_contiguous, unsigned long long *d_results, int n) {
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n)
		return;
	d_results[idx] = compactKernel(d_contiguous + (idx * PIXELS));
}

std::vector<unsigned long long> batchCompact(thrust::device_vector<bool> d_contiguous) {
	size_t size = d_contiguous.size() / PIXELS;
	thrust::device_vector<unsigned long long> d_results(size);

	// Cast to pointers
	bool* d_contiguousPtr = thrust::raw_pointer_cast(&d_contiguous[0]);
	unsigned long long* d_resultsPtr = thrust::raw_pointer_cast(&d_results[0]);

	// Compact
	size_t blocks = ceil(size / THREADS) + 1;
	compactBatchKernel << <blocks, THREADS >> > (d_contiguousPtr, d_resultsPtr, size);

	// Results
	thrust::host_vector<unsigned long long> h_results = d_results;
	return std::vector<unsigned long long>(h_results.begin(), h_results.end());
}
/*
void test() {
	thrust::device_vector<bool> contiguous(64 * 2);
	for (int i = 0; i < PIXELS; i++)
		contiguous[i] = i % 2;
	for (int i = PIXELS; i < PIXELS * 2; i++)
		contiguous[i] = !(i % 2);

	std::vector<unsigned long long> results = batchCompact(contiguous);
	for (int i = 0; i < results.size(); i++) {
		printf("%d\n", results[i]);
		std::cout << std::bitset<64>(results[i]) << std::endl;
	}
}*/