#include "CudaResize.cuh"

#include<vector>

#include <cuda_runtime.h>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/gather.h>
#include<thrust/iterator/counting_iterator.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>

#include"HostUtils.h"

__global__ void downSampleKernel(unsigned char * d_in, unsigned char * d_out, size_t skip) {
	size_t i = threadIdx.x;
	// Assuming 3 channels BGR and averaging
	int px = d_in[i * skip * 3] + d_in[i * skip * 3 + 1] + d_in[i * skip * 3 + 2];
	d_out[i] = px/3;
}

std::vector<unsigned char> resizeImageDevice(cv::Mat h_mat) {
	// Format 8b BGR BGR BGR
	h_mat.convertTo(h_mat, CV_8U);

	// Get total file size
	size_t size = h_mat.size().area() * h_mat.channels();

	// Copy the data
	unsigned char *ptr = (unsigned char*)h_mat.data;
	thrust::host_vector<unsigned char> h_img(ptr, ptr + size);

	// Copy to device
	thrust::device_vector<unsigned char> d_img = h_img;
	thrust::device_vector<unsigned char> d_results(PIXELS);

	// Gather required pixels
	size_t skip = size / PIXELS / h_mat.channels();
	assert(skip > 0);
	unsigned char * d_img_ptr = thrust::raw_pointer_cast(&d_img[0]);
	unsigned char * d_results_ptr = thrust::raw_pointer_cast(&d_results[0]);
	downSampleKernel << <1, PIXELS >> > (d_img_ptr, d_results_ptr, skip);

	// Copy back to host
	thrust::host_vector<unsigned char> h_results = d_results;
	std::vector<unsigned char> temp = std::vector<unsigned char>(h_results.begin(), h_results.end());

	//cv::Mat result = cv::Mat(8,8,CV_8UC1,(void *)&temp[0],8);
	//imwrite("C:\\Users\\windows\\Desktop\\New folder\\1.png", result);
	//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	//imshow("Display window", result);
	//cv::waitKey(0);

	return temp;
}

std::vector<unsigned char> loadAllFilesDevice(std::vector<std::string> fileNames) {
	std::vector<unsigned char> images;
	images.reserve(fileNames.size() * PIXELS);
	for (int i = 0; i < fileNames.size(); i++) {
		std::vector<unsigned char> image = resizeImageDevice(cv::imread(fileNames[i]));
		printf("Loaded: %s\n", fileNames[i].c_str());
		images.insert(images.end(), image.begin(), image.end());
	}
	return images;
}