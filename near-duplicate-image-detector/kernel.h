#pragma once
#include <thrust/host_vector.h>

#define SIZE 8
#define PIXELS (SIZE * SIZE)

unsigned long sumAbsoluteDifference(thrust::host_vector<char> h_a, thrust::host_vector<char> h_b);
unsigned long aHash(thrust::host_vector<unsigned char> h_img);
//unsigned long long boolVectorToLongCpu(thrust::host_vector<bool> arr);
