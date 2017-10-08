#pragma once

#include<thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/system/cuda/execution_policy.h>

unsigned long long dHash(thrust::host_vector<unsigned char> h_img);
std::vector<unsigned long long> dHashBatch(std::vector<unsigned char> v_imgs,size_t batchSize = 2);
//void test();