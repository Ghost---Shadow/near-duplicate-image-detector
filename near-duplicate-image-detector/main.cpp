#include<iostream>
#include<vector>
#include<string>
#include<cstdio>
#include<fstream>

#include"HostUtils.h"
#include"HostHashing.h"

#include"dHashDevice.cuh"
#include"DeviceUtils.cuh"

int main(void) {
	std::string path = "./images/";
	//std::string path = "E:/The Vampire Diaries/New folder/000-099/";

	// File names
	printf("Getting list of files\n");
	std::vector<std::string> fileNames = listFiles(path);

	// Load all files
	printf("Loading files\n");
	std::vector<unsigned char> images;
	//images = loadAllFilesDevice(fileNames,1);

	for (int i = 0; i < fileNames.size(); i++) {
		std::vector<unsigned char> image = loadImage(fileNames[i]);
		images.insert(images.end(), image.begin(), image.end());
	}
	printf("Files loaded\n");
	printf("Total images: %d\n", images.size() / PIXELS);

	// Compute hashes
	printf("Computing hashes\n");
	size_t batch_size = 10;
	std::vector<unsigned long long> hashes = dHashBatch(images, batch_size);

	printf("Hashes computed %d\n", hashes.size());

	for (int i = 0; i < images.size() / PIXELS; i++) {
		auto start = images.begin() + (i * PIXELS);
		std::vector<unsigned char> img(start, start + PIXELS);
		unsigned long long host_hash = dHashHost(img);
		printf("%llx - %llx = %llx\n", hashes[i], host_hash, hammingDistanceHost(hashes[i], host_hash));
	}

	// Print hamming distances
	printf("Calculating hamming distances\n");
	dumpJson(path, "list.json", fileNames, hashes);
	printf("Written into json file\n");

	system("pause");
	return 0;
}