#include<iostream>
#include<vector>
#include<string>
#include<cstdio>

#include<opencv2\core.hpp>
#include<opencv2\opencv.hpp>

#include <thrust/host_vector.h>

#include"HostUtils.h"

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

std::vector<unsigned char> loadImage(std::string fileName, bool gray) {
	// Read image to mat
	cv::Mat img = cv::imread(fileName);
	// Format 8b BGR BGR BGR
	img.convertTo(img, CV_8U);

	// To Gray
	if (gray)
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

	// Resize to nxn
	cv::resize(img, img, { SIZE,SIZE });

	// Get total file size
	size_t size = img.size().area() * img.channels();
	printf("%s %d bytes\n", fileName.c_str(), size);

	// Copy the data
	unsigned unsigned char *ptr = (unsigned char*)img.data;
	std::vector<unsigned char> temp(ptr, ptr + size);

	/*
	for (int i = 0; i < 65; i++) {
	printf("%d\t", ptr[i]);
	}
	printf("\n");
	*/

	return temp;
}

unsigned char hammingDistance(const unsigned long long &a, const unsigned long long &b) {
	unsigned long long c = a ^ b;
	unsigned char result = 0;

	while (c) {
		result += c & 1;
		c >>= 1;
	}

	return result;
}