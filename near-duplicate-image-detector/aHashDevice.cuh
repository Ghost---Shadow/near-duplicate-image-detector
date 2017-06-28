#pragma once

#include<thrust/host_vector.h>

unsigned long long aHash(thrust::host_vector<unsigned char> h_img);
std::vector<unsigned long long> aHashBatch(thrust::host_vector<unsigned char> h_imgs);