#pragma once
#include<vector>

#include<opencv2\core.hpp>

std::vector<unsigned char> resizeImageDevice(cv::Mat h_img);
std::vector<unsigned char> loadAllFilesDevice(std::vector<std::string> fileNames);