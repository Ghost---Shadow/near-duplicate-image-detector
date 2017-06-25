#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include<thrust/functional.h>

#include <iostream>
#include<bitset>

#include"kernel.h"
#include"HostUtils.h"

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

template <typename T>
struct isGreaterFunctor
{
	__host__ __device__
		float operator()(const T& x, const T& y) const {
		return x > y ? 1 : 0;
	}
};

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

	// Compact on CPU
	return boolVectorToLongCpu(uncompacted);
}

unsigned long long dHash(thrust::host_vector<unsigned char> h_img) {
	// Copy image to device
	thrust::device_vector<unsigned char> d_img = h_img;

	// Allocate space for storing results
	thrust::device_vector<bool> uncompacted(PIXELS);

	// Calculate gradient
	thrust::transform(d_img.begin(), d_img.end(), d_img.begin() + 1, uncompacted.begin(), isGreaterFunctor<unsigned char>());

	// Compact on CPU
	return boolVectorToLongCpu(uncompacted);
}

std::vector<unsigned long long> dHashBatch(thrust::host_vector < thrust::host_vector<unsigned char>> h_imgs) {
	// Uncompacted results
	thrust::device_vector<thrust::device_vector<bool>> uncompacted(h_imgs.size());

	// Calculate uncompacted gradients


	// Compact

	// Results
	thrust::host_vector<unsigned long long> h_results;	
	return std::vector<unsigned long long>(h_results.begin(), h_results.end());
}