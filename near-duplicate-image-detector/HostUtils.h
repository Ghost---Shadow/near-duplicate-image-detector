#pragma once

#define SIZE 8
#define PIXELS (SIZE * SIZE)

unsigned long long compactHost(std::vector<bool> arr);
std::vector<unsigned char> loadImage(std::string fileName, bool gray = true);
unsigned char hammingDistanceHost(const unsigned long long &a, const unsigned long long &b);
std::vector<std::string> listFiles(std::string path);
void dumpJson(std::string path,	std::string jsonName, std::vector<std::string> fileNames, std::vector<unsigned long long> hashes);