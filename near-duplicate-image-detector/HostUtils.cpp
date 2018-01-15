#include<iostream>
#include<vector>
#include<string>
#include<cstdio>
#include <filesystem>
#include<bitset>

#include<opencv2\core.hpp>
#include<opencv2\opencv.hpp>

#include"HostUtils.h"
#include"DeviceUtils.cuh"

namespace fs = std::experimental::filesystem;

unsigned long long compactHost(std::vector<bool> arr) {
	unsigned long long result = 0;

	assert(arr.size() == PIXELS);

	unsigned long long temp;
	for (int i = 0; i < PIXELS; i++) {
		temp = arr[i];
		result |= temp << (PIXELS - i - 1);
	}
	return result;
}

std::vector<std::string> listFiles(std::string path) {
	std::vector<std::string> fileNames;
	for (auto & p : fs::recursive_directory_iterator(path)) {
		std::string fileName = p.path().string();
		std::string copy = fileName;
		std::string extention = copy.substr(fileName.size() - 4);
		if (extention == ".jpg" || extention == ".png") {
			//std::cout << fileName << std::endl;
			fileNames.push_back(fileName);
		}
	}
	return fileNames;
}

void dumpJson(std::string path,
	std::string jsonName,
	std::vector<std::string> fileNames,
	std::vector<unsigned long long> hashes) {

	std::ofstream handle(path+jsonName);
	handle << "{\n\t\"images\":[\n";
	for (int i = 0; i < fileNames.size(); i++) {
		std::vector<unsigned char> distances = batchHamming(i, hashes);
		std::string imageName = fileNames[i].substr(path.length());
		std::replace(imageName.begin(), imageName.end(), '\\', '/');
		handle << "\t\t[\"" << imageName << "\",[";
		for (int j = 0; j < distances.size(); j++) {
			handle << int(distances[j]);
			if (j < distances.size() - 1)
				handle << ",";
		}
		handle << "]]";
		if (i < fileNames.size() - 1)
			handle << ",";
		handle << "\n";
	}
	handle << "\t]\n}";
	handle.close();
}

std::vector<unsigned char> loadImageOpenCv(std::string fileName, bool gray) {
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
	printf("%s\t%d bytes\n", fileName.c_str(), size);

	// Copy the data
	unsigned unsigned char *ptr = (unsigned char*)img.data;
	std::vector<unsigned char> temp(ptr, ptr + size);

	return temp;
}

std::vector<unsigned char> loadImage(std::string fileName, bool gray) {
	// Read image to mat
	cv::Mat img = cv::imread(fileName);
	// Format 8b BGR BGR BGR
	img.convertTo(img, CV_8U);
	
	// Get total file size
	size_t size = img.size().area() * img.channels();
	printf("%s\t%d bytes\n", fileName.c_str(), size);

	// Copy the data
	unsigned char *ptr = (unsigned char*)img.data;
	std::vector<unsigned char> temp(PIXELS);
	size_t skip = size / PIXELS;
	for (int i = 0; i < PIXELS; i++) {
		size_t offset = i * skip;
		temp[i] = (ptr[offset] + ptr[offset + 1] + ptr[offset + 2]) / 3;
	}

	return temp;
}

unsigned char hammingDistanceHost(const unsigned long long &a, const unsigned long long &b) {
	unsigned long long x = a ^ b;
	std::bitset<64> bx(x);
	return bx.count();
}