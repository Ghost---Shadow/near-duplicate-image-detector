#pragma once
#include<vector>
#include<thrust/device_vector.h>

#include<opencv2\core.hpp>

std::vector<unsigned char> resizeImageDevice(cv::Mat h_img);
std::vector<unsigned char> loadAllFilesDevice(std::vector<std::string> fileNames,size_t batchSize = 10);

void loadImagesToHost(const std::vector<std::string> fileNames,
	std::vector<unsigned char> &h_imgs,
	thrust::device_vector<size_t> &d_skip,
	thrust::device_vector<size_t> &d_sizes);

void resizeImageBatchDevice(const std::vector<unsigned char> &h_imgs,
	const thrust::device_vector<size_t> &d_skip,
	const thrust::device_vector<size_t> &d_sizes,
	thrust::device_vector<unsigned char> &d_out);