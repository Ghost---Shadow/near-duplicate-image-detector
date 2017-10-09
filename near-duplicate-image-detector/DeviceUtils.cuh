#pragma once

#include<thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/system/cuda/execution_policy.h>

#define THREADS 32

void batchCompact(const thrust::device_vector<bool> d_contiguous,
	thrust::host_vector <unsigned long long,
	thrust::cuda::experimental::pinned_allocator<unsigned long long>> &h_contiguous,
	cudaStream_t &s);

std::vector<unsigned char> batchHamming(size_t base, thrust::device_vector<unsigned long long> hashes);
void test();