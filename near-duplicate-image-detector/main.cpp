
#include<iostream>
#include<vector>
#include<string>
#include<cstdio>

#include<opencv2\core.hpp>
#include<opencv2\opencv.hpp>

#include"kernel.h"

std::vector<unsigned char> loadImage(std::string fileName, bool gray = true) {
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

int main(void) {	
	// File names
	std::vector<std::string> fileNames = { "c1.jpg","c2.jpg","c3.jpg","c4.jpg" };

	// Container for images
	std::vector<std::vector<unsigned char>> images;

	// Load all files
	for (int i = 0; i < fileNames.size(); i++) {
		images.push_back(loadImage(fileNames[i]));
	}

	//long result = sumAbsoluteDifference(images[0], images[1]);

	//std::cout << result << std::endl;
	std::vector<unsigned long long> hashes;
	for (int i = 0; i < images.size(); i++) {
		unsigned long long hash = dHash(images[i]);
		hashes.push_back(hash);
		printf("%llx\n", hash);
	}
	
	system("pause");
	return 0;
}