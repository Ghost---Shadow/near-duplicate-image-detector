#pragma once

#include<thrust/host_vector.h>

unsigned long long dHash(thrust::host_vector<unsigned char> h_img);
std::vector<unsigned long long> dHashBatch(thrust::host_vector <unsigned char> h_imgs);
//void test();