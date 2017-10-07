#include "CudaResize.cuh"

#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include"HostUtils.h"

std::vector<unsigned char> resizeImageDevice(cv::Mat h_img) {
	cv::cuda::GpuMat d_img;
	d_img.upload(h_img);

	// Resize to nxn	
	//cv::cuda::resize(d_img, d_img, { SIZE,SIZE }); // NOT WORKING
	
	d_img.download(h_img);

	// Get total file size
	size_t size = h_img.size().area() * h_img.channels();
	//printf("%s\t%d bytes\n", fileName.c_str(), size);

	// Copy the data
	unsigned char *ptr = (unsigned char*)h_img.data;
	std::vector<unsigned char> temp(ptr, ptr + size);

	return temp;
}

std::vector<unsigned char> loadAllFilesDevice(std::vector<std::string> fileNames)
{
	std::vector<unsigned char> images;
	images.reserve(fileNames.size() * PIXELS);
	for (int i = 0; i < fileNames.size(); i++) {
		std::vector<unsigned char> image = resizeImageDevice(loadImage(fileNames[i]));
		printf("Loaded: %s\n", fileNames[i].c_str());
		images.insert(images.end(), image.begin(), image.end());
	}
	return images;
}