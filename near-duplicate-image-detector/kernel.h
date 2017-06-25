#pragma once
#include <thrust/host_vector.h>

unsigned long long sumAbsoluteDifference(thrust::host_vector<char> h_a, thrust::host_vector<char> h_b);
unsigned long long aHash(thrust::host_vector<unsigned char> h_img);
unsigned long long dHash(thrust::host_vector<unsigned char> h_img);
