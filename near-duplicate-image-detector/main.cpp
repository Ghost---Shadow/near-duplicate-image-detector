#include<iostream>
#include<vector>
#include<string>
#include<cstdio>

#include"kernel.h"
#include"HostUtils.h"

int main(void) {	
	std::string path = "./";

	// File names
	std::vector<std::string> fileNames = listFiles(path);

	// Container for images
	std::vector<std::vector<unsigned char>> images;

	// Load all files
	for (int i = 0; i < fileNames.size(); i++) {
		images.push_back(loadImage(fileNames[i]));
	}

	// Compute hashes
	std::vector<unsigned long long> hashes;
	for (int i = 0; i < images.size(); i++) {
		unsigned long long hash = dHash(images[i]);
		hashes.push_back(hash);
		printf("%llx\n", hash);
	}

	// Print hamming distances
	for (int i = 0; i < hashes.size(); i++) {
		for (int j = i + 1; j < hashes.size(); j++) {
			printf("%d\t", hammingDistance(hashes[i], hashes[j]));
		}
		printf("\n");
	}
	
	system("pause");
	return 0;
}