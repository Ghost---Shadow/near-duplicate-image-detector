#include<iostream>
#include<vector>
#include<string>
#include<cstdio>

#include"kernel.h"
#include"HostUtils.h"

int main(void) {	
	// File names
	std::vector<std::string> fileNames = { "c1.jpg","c2.jpg","c3.jpg"};

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