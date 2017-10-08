#include<iostream>
#include<vector>
#include<string>
#include<cstdio>
#include<fstream>

#include"HostUtils.h"
#include"HostHashing.h"

#include"aHashDevice.cuh"
#include"dHashDevice.cuh"
#include"DeviceUtils.cuh"
#include"CudaResize.cuh"

int main(void) {	
	std::string path = "./images/";

	// File names
	printf("Getting list of files\n");
	std::vector<std::string> fileNames = listFiles(path);
	
	// Load all files
	printf("Loading files\n");
	std::vector<unsigned char> images = loadAllFilesDevice(fileNames);
	/*for (int i = 0; i < fileNames.size(); i++) {
		std::vector<unsigned char> image = loadImage(fileNames[i]);
		images.insert(images.end(), image.begin(), image.end());
	}*/	
	printf("Files loaded\n");
	printf("Total images: %d\n", images.size() / PIXELS);
	
	// Compute hashes
	printf("Computing hashes\n");
	std::vector<unsigned long long> hashes = dHashBatch(images);
	printf("Hashes computed %d\n",hashes.size());

	for (int i = 0; i < images.size()/PIXELS; i++) {
		auto start = images.begin() + (i * PIXELS);
		std::vector<unsigned char> img(start, start + PIXELS);
		printf("%llx %llx\n", hashes[i],dHashHost(img));
		//printf("%llx %llx\n",aHashHost(img),dHashHost(img));
	}
	
	// Print hamming distances
	printf("Calculating hamming distances\n");
	dumpJson(path,"list.json",fileNames,hashes);
	printf("Written into json file\n");
	
	system("pause");
	return 0;
}