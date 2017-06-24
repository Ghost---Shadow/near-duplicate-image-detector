
#include<iostream>
#include<vector>
#include<string>
#include<cstdio>

#include<opencv2\core.hpp>
#include<opencv2\opencv.hpp>

#include"kernel.h"

std::vector<unsigned char> loadImage(std::string fileName) {
	// Read image to mat
	cv::Mat img = cv::imread(fileName);
	// Format 8b BGR BGR BGR
	img.convertTo(img, CV_8U);

	// Get total file size
	size_t size = img.size().area() * img.channels();
	printf("%s %d bytes\n", fileName.c_str(), size);

	// Copy the data
	unsigned unsigned char *ptr = (unsigned char*)img.data;
	std::vector<unsigned char> temp(ptr, ptr + size);

	/*
	for (int i = 0; i < 10; i++) {
		printf("%d\t", ptr[i]);
	}
	printf("\n");
	*/

	return temp;
}

int main(void)
{
	// File names
	std::vector<std::string> fileNames = { "t1.png","t2.png" };
	
	// Container for images
	std::vector<std::vector<unsigned char>> images;

	// Load all files
	for (int i = 0; i < fileNames.size(); i++)
		images.push_back(loadImage(fileNames[i]));	

	long result = sumAbsoluteDifference(images[0], images[1]);

	std::cout << result << std::endl;

	system("pause");
	return 0;
}