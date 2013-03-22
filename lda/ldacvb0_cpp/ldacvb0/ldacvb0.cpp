/**
	@file
	@brief LDA CVB0

	Copyright (C) 2013 Nakatani Shuyo / Cybozu Labs, Inc., all rights reserved.
*/

#include <iostream>
#include <fstream>
#include "cybozu/string.hpp"
#include "ldacvb0.hpp"


int main(int argc, char* argv[]) {
	std::string filename = "C:/works/iir/lda/ali.txt";
	std::ifstream ifs(filename, std::ios::binary);
	while (ifs) {
		std::string st;
		ifs >> st;
		std::cout << st << "/";
	}
	std::cout << std::endl;

	return 0;
}
