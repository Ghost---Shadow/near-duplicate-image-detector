#pragma once

#define SIZE 8
#define PIXELS (SIZE * SIZE)

unsigned long long boolVectorToLongCpu(thrust::host_vector<bool> arr);
std::vector<unsigned char> loadImage(std::string fileName, bool gray = true);
unsigned char hammingDistance(const unsigned long long &a, const unsigned long long &b);