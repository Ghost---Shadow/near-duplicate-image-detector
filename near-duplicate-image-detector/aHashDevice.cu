#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

#include "aHashDevice.cuh"
#include "HostUtils.h"
#include "DeviceUtils.cuh"

template <typename T>
struct isGreaterThanAvg {
	T avg;
	isGreaterThanAvg(T avg) : avg(avg) {}

	__host__ __device__
		bool operator()(const T &value) const {
		return value > avg;
	}
};

unsigned long long aHash(thrust::host_vector<unsigned char> h_img) {
	// Copy image to device
	thrust::device_vector<unsigned char> d_img = h_img;

	// Calculate average
	unsigned char average = thrust::reduce(d_img.begin(), d_img.end(), (unsigned long)0, thrust::plus<unsigned long>()) / PIXELS;

	// Allocate space for storing results
	thrust::device_vector<bool> uncompacted(PIXELS);

	// Set all the pixels greater than average
	thrust::transform(d_img.begin(), d_img.end(), uncompacted.begin(), isGreaterThanAvg<unsigned char>(average));

	// Compact on CPU
	return compactCpu(uncompacted);
}

std::vector<unsigned long long> aHashBatch(thrust::host_vector<unsigned char> h_imgs) {
	return std::vector<unsigned long long>();
}