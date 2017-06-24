
#include<iostream>
#include<vector>

#include"kernel.h"

int main(void)
{
	std::vector<char> a = { 1,2,3,4,5 };
	std::vector<char> b = { 5,4,3,2,1 };

	long result = getDisimilarity(a, b);

	std::cout << result << std::endl;

	system("pause");
	return 0;
}