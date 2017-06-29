#include <numeric>
#include <algorithm>

#include "HostHashing.h"
#include "HostUtils.h"

template <typename T>
struct aHashFunctor {
	T avg;
	aHashFunctor(T avg) : avg(avg) {}

	bool operator()(const T &value) const {
		return value > avg;
	}
};

template <typename T>
struct dHashFunctor {
	bool operator()(const T &x, const T &y) const {
		return x > y;
	}
};

unsigned long long aHashHost(std::vector<unsigned char> img) {
	// Calculate average
	unsigned char average = std::accumulate(img.begin(), img.end(),0) / PIXELS;

	// Calculate aHash
	std::vector<char> uncompacted(PIXELS);
	std::transform(img.begin(), img.end(), uncompacted.begin(), aHashFunctor<unsigned char>(average));

	// Compact bool array to unsigned long long
	std::vector<bool> bUncompacted(uncompacted.begin(), uncompacted.end());
	return compactHost(bUncompacted);
}

unsigned long long dHashHost(std::vector<unsigned char> img) {
	// Calculate dHash
	std::vector<char> uncompacted(PIXELS);
	std::adjacent_difference(img.begin(), img.end(), uncompacted.begin(), dHashFunctor<unsigned char>());
	
	// Compact bool array to unsigned long long
	std::vector<bool> bUncompacted(uncompacted.begin(), uncompacted.end());
	return compactHost(bUncompacted);
}
