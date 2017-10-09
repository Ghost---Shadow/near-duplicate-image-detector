#include<iostream>
#include<bitset>

#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

#include <cuda_runtime.h>
#include"device_functions.h"
#include"cuda.h"

#include"DeviceUtils.cuh"
#include"HostUtils.h"

__device__ unsigned long long compactKernel(const bool * arr) {
	unsigned long long result = 0;
	unsigned long long temp;
	for (int i = 0; i < PIXELS; i++) {
		temp = arr[i];
		result |= temp << (PIXELS - i - 1);
	}
	return result;
}

__global__ void compactBatchKernel(const bool * d_contiguous, unsigned long long *d_results,const int n) {
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n)
		return;
	d_results[idx] = compactKernel(d_contiguous + (idx * PIXELS));
}

struct hammingFunctor {
	__host__ __device__
		unsigned char operator()(const unsigned long long &a, const unsigned long long &b) const {
		unsigned long long c = a ^ b;
		unsigned char result = 0;

		while (c) {
			c &= (c - 1);
			result++;
		}

		return result;
	}
};

void batchCompact(const thrust::device_vector<bool> d_contiguous,
	thrust::host_vector <unsigned long long,
	thrust::cuda::experimental::pinned_allocator<unsigned long long>> &h_contiguous,
	cudaStream_t &s) {
	
	size_t size = d_contiguous.size() / PIXELS;
	thrust::device_vector<unsigned long long> d_results(size);

	// Cast to pointers
	const bool* d_contiguousPtr = thrust::raw_pointer_cast(d_contiguous.data());
	unsigned long long* d_resultsPtr = thrust::raw_pointer_cast(d_results.data());

	// Compact
	size_t blocks = ceil(size / THREADS) + 1;
	compactBatchKernel << <blocks, THREADS >> > (d_contiguousPtr, d_resultsPtr, size);

	// Results
	cudaMemcpyAsync(thrust::raw_pointer_cast(h_contiguous.data()), d_resultsPtr, d_results.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost, s);
}

std::vector<unsigned char> batchHamming(size_t base, thrust::device_vector<unsigned long long> d_hashes) {
	thrust::device_vector<unsigned char> d_distances(d_hashes.size());
	thrust::device_vector<unsigned long long> d_base(d_hashes.size());
	thrust::fill(d_base.begin(), d_base.end(), d_hashes[base]);

	// Compute hamming distances
	thrust::transform(d_hashes.begin(), d_hashes.end(), d_base.begin(), d_distances.begin(), hammingFunctor());

	thrust::host_vector<unsigned char> h_distances = d_distances;
	return std::vector<unsigned char>(h_distances.begin(), h_distances.end());
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

/*
void test() {
	std::vector<unsigned long long> t(3);
	t[0] = 0x8004022a17938700;
	t[1] = 0x322454546478e860;
	t[2] = 0xc11a2abad1baa851;

	std::vector<unsigned char> result = batchHamming(1, t);
	for (int i = 0; i < 3; i++) {
		printf("%d ", result[i]);
	}
	printf("\n");
}*/