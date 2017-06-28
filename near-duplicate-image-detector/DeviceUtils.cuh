#pragma once

#include<thrust/device_vector.h>

#define THREADS 32

std::vector<unsigned long long> batchCompact(thrust::device_vector<bool> contiguous);