#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/adjacent_difference.h>

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
	thrust::device_vector<bool> uncompacted(PIXELS);

	// Calculate gradient
	thrust::adjacent_difference(d_img.begin(), d_img.end(), uncompacted.begin(), isGreaterFunctor<unsigned char>());

	// Compact on CPU
	return compactHost(uncompacted);
}

std::vector<unsigned long long> dHashBatch(thrust::host_vector <unsigned char> h_imgs) {
	// Copy image to device
	thrust::device_vector<unsigned char> d_imgs = h_imgs;

	// Uncompacted results
	thrust::device_vector<bool> uncompacted(h_imgs.size());

	// Calculate gradient
	thrust::adjacent_difference(d_imgs.begin(), d_imgs.end(), uncompacted.begin(), isGreaterFunctor<unsigned char>());

	// Return the compacted batch
	return batchCompact(uncompacted);
}
