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
	d_out[i] = px / 3;
}

__global__ void downSampleBatchKernel(const unsigned char * d_in,
	unsigned char * d_out,
	const size_t * skipArray,
	const size_t * base)
{
	size_t i = threadIdx.x;
	size_t j = blockIdx.x;
	size_t skip = skipArray[j];
	size_t offset = base[j];
	int px = d_in[offset + i * skip * 3] + d_in[offset + i * skip * 3 + 1] + d_in[offset + i * skip * 3 + 2];
	d_out[j * PIXELS + i] = px / 3;
}

void loadImagesToHost(const std::vector<std::string> fileNames,
	std::vector<unsigned char> &h_imgs,
	thrust::device_vector<size_t> &d_skip,
	thrust::device_vector<size_t> &d_sizes) {

	// Clear existing data
	d_skip.clear();
	d_sizes.clear();
	assert(d_skip.capacity() == fileNames.size());
	assert(d_sizes.capacity() == fileNames.size() + 1); 

	// Bases are sizes shifted by one
	d_sizes.push_back(0);

	for (int i = 0; i < fileNames.size(); i++) {
		// Read image from disk
		cv::Mat h_mat = cv::imread(fileNames[i]);
		h_mat.convertTo(h_mat, CV_8U);
		std::cout << "Loaded: " << fileNames[i] << std::endl;

		// Calculate and store size
		size_t size = h_mat.size().area() * h_mat.channels();
		d_sizes.push_back(size);

		// Append image to array
		unsigned char *ptr = (unsigned char*)h_mat.data;
		h_imgs.insert(h_imgs.end(), ptr, ptr + size);

		// Append skip to array
		d_skip.push_back(size / PIXELS / h_mat.channels());
	}
	// Drop the last size
	d_sizes.pop_back();
	std::cout << "Resizing" << std::endl;
}

void resizeImageBatchDevice(const std::vector<unsigned char> &h_imgs,
	const thrust::device_vector<size_t> &d_skip,
	const thrust::device_vector<size_t> &d_sizes,
	thrust::device_vector<unsigned char> &d_out) {

	size_t imageCount = d_sizes.size();

	// Load all data to Device
	std::cout << "Allocating space\n";
	thrust::device_vector<unsigned char> d_imgs(h_imgs.begin(), h_imgs.end());
	std::cout << "Space allocation done\n";

	// Cast pointers
	unsigned char * d_imgs_ptr = thrust::raw_pointer_cast(&d_imgs[0]);
	unsigned char * d_out_ptr = thrust::raw_pointer_cast(&d_out[0]);
	const size_t * d_skip_ptr = thrust::raw_pointer_cast(&d_skip[0]);
	const size_t * d_sizes_ptr = thrust::raw_pointer_cast(&d_sizes[0]);

	// Call the kernel
	downSampleBatchKernel << <imageCount, PIXELS >> > (d_imgs_ptr, d_out_ptr, d_skip_ptr, d_sizes_ptr);
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

// TODO: (Optimization) Do not send the resized images to host
std::vector<unsigned char> loadAllFilesDevice(std::vector<std::string> fileNames, size_t batchSize) {
	/*std::vector<unsigned char> images;
	images.reserve(fileNames.size() * PIXELS);
	for (int i = 0; i < fileNames.size(); i++) {
		std::vector<unsigned char> image = resizeImageDevice(cv::imread(fileNames[i]));
		printf("Loaded: %s\n", fileNames[i].c_str());
		images.insert(images.end(), image.begin(), image.end());
	}
	return images;*/

	std::vector<unsigned char> h_out;
	thrust::device_vector<size_t> d_skip(batchSize), d_sizes(batchSize+1);
	thrust::device_vector<unsigned char> d_out(batchSize * PIXELS);

	// TODO: FIX batchSize
	for (int i = 0; i < fileNames.size(); i += batchSize) {
		std::vector<unsigned char> h_imgs;
		std::vector<size_t> h_skip;
		std::vector<size_t> h_sizes;

		std::vector<std::string> batchFiles(fileNames.begin() + i, fileNames.begin() + i + batchSize);

		loadImagesToHost(batchFiles, h_imgs, d_skip, d_sizes);
		resizeImageBatchDevice(h_imgs, d_skip, d_sizes, d_out);
		h_out.insert(h_out.end(), d_out.begin(), d_out.end());
	}
	return h_out;
}