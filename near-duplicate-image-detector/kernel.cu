#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include<thrust/functional.h>

#include <iostream>

#include"kernel.h"

struct absdiff
{
	__host__ __device__
		float operator()(const char& x, const char& y) const
	{
		return abs(x - y);
	}
};

long sumAbsoluteDifference(thrust::host_vector<char> h_a, thrust::host_vector<char> h_b) {
	thrust::device_vector<char> d_a = h_a;
	thrust::device_vector<char> d_b = h_b;

	thrust::device_vector<char> d_res(h_a.size());

	thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_res.begin(), absdiff());
	long sum = thrust::reduce(d_res.begin(), d_res.end(), (long) 0 , thrust::plus<long>());

	return sum;
}