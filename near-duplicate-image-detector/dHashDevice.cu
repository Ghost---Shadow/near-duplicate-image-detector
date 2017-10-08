#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/adjacent_difference.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/system/cuda/execution_policy.h>

#include"dHashDevice.cuh"
#include"HostUtils.h"
#include"DeviceUtils.cuh"

template <typename T>
struct isGreaterFunctor {
	__host__ __device__
		bool operator()(const T& x, const T& y) const {
		return x > y;
	}
};

unsigned long long dHash(thrust::host_vector<unsigned char> h_img) {
	// Copy image to device
	thrust::device_vector<unsigned char> d_img = h_img;

	// Allocate space for storing results
	thrust::device_vector<bool> d_uncompacted(PIXELS);

	// Calculate gradient
	thrust::adjacent_difference(d_img.begin(), d_img.end(), d_uncompacted.begin(), isGreaterFunctor<unsigned char>());

	// Compact on CPU
	thrust::host_vector<bool> h_uncompacted = d_uncompacted;
	return compactHost(std::vector<bool>(h_uncompacted.begin(), h_uncompacted.end()));
}

std::vector<unsigned long long> dHashBatch(std::vector<unsigned char> v_imgs, size_t batchSize) {
	thrust::host_vector <unsigned char,
		thrust::cuda::experimental::pinned_allocator<unsigned char>> h_imgs(batchSize * PIXELS);

	size_t imageCount = v_imgs.size() / PIXELS;

	// Stream data to device
	cudaStream_t s;
	cudaStreamCreate(&s);
	thrust::device_vector<unsigned char> d_imgs(h_imgs.size());
	for (int i = 0; i < imageCount; i += batchSize) {
		thrust::copy(v_imgs.begin() + i * PIXELS, v_imgs.begin() + (i + batchSize) * PIXELS , h_imgs.begin());
		cudaMemcpyAsync(thrust::raw_pointer_cast(d_imgs.data()), thrust::raw_pointer_cast(h_imgs.data()), d_imgs.size() * sizeof(unsigned char), cudaMemcpyHostToDevice, s);

		// Uncompacted results
		thrust::device_vector<bool> d_uncompacted(h_imgs.size());

		// Calculate gradient
		thrust::adjacent_difference(thrust::cuda::par.on(s), d_imgs.begin(), d_imgs.end(), d_uncompacted.begin(), isGreaterFunctor<unsigned char>());

		// TODO: use same stream
		cudaStreamSynchronize(s);
	}
	cudaStreamDestroy(s);

	// Return the compacted batch
	return batchCompact(d_uncompacted);
	//return std::vector<unsigned long long>();
}
