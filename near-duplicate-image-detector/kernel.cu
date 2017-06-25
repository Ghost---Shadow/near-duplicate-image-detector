#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include<thrust/functional.h>

#include <iostream>
#include<bitset>

#include"kernel.h"

struct absdiff
{
	__host__ __device__
		float operator()(unsigned const char& x, unsigned const char& y) const {
		return abs(x - y);
	}
};

template <typename T>
struct isGreaterThanAvg
{
	T avg;
	isGreaterThanAvg(T avg) : avg(avg) {}

	__host__ __device__
		bool operator()(const T &value) const {
		return value > avg ? 1 : 0;
	}
};

unsigned long long boolVectorToLongCpu(thrust::host_vector<bool> arr) {
	unsigned long long result = 0;

	assert(arr.size() == PIXELS);

	unsigned long long temp;
	for (int i = 0; i < PIXELS; i++) {
		temp = arr[i];
		result |= temp << (PIXELS - i - 1);
	}
	return result;
}

unsigned long long sumAbsoluteDifference(thrust::host_vector<unsigned char> h_a, thrust::host_vector<unsigned char> h_b) {
	// Copy images to device
	thrust::device_vector<unsigned char> d_a = h_a;
	thrust::device_vector<unsigned char> d_b = h_b;

	// Allocate space for result array
	thrust::device_vector<unsigned char> d_res(h_a.size());

	// Find absolute difference
	thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_res.begin(), absdiff());

	// Find summation of absolute difference
	unsigned long long sum = thrust::reduce(d_res.begin(), d_res.end(), (unsigned long)0, thrust::plus< unsigned long>());

	return sum;
}

unsigned long long aHash(thrust::host_vector<unsigned char> h_img) {
	// Copy image to device
	thrust::device_vector<unsigned char> d_img = h_img;

	// Calculate average
	unsigned char average = thrust::reduce(d_img.begin(), d_img.end(), (unsigned long)0, thrust::plus<unsigned long>()) / PIXELS;

	// Allocate space for storing results
	thrust::device_vector<bool> uncompacted(PIXELS);

	// Set all the pixels greater than average
	thrust::transform(d_img.begin(), d_img.end(), uncompacted.begin(), isGreaterThanAvg<unsigned char>(average));

	// Compact on CPU because it is only 64 iterations
	return boolVectorToLongCpu(uncompacted);
}