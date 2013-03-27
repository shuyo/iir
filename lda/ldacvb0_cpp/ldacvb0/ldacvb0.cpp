/**
	@file
	@brief LDA CVB0
	Latent Dirichlet Allocation - Collapsed Variational Bayesian Estimation

	Copyright (C) 2013 Nakatani Shuyo / Cybozu Labs, Inc., all rights reserved.
	This code is licensed under the MIT license.
*/

#include <iostream>
#include <fstream>
#include "cybozu/string.hpp"
#include "ldacvb0.hpp"


int main(int argc, char* argv[]) {
	//std::string filename = "C:/works/iir/lda/ali.txt";

	cybozu::ldacvb0::Documents<std::string, char> docs;

	for(int i=0;i<argc;++i) {
		std::string filename(argv[i]);


		try {
			cybozu::Mmap map(filename);
			const char *p = map.get();
			const char *end = p + map.size();
			docs.add(p, end);

		} catch (std::exception& e) {
			printf("%s\n", e.what());
		}
	}

/*	std::ifstream ifs(filename, std::ios::binary);
	while (ifs) {
		std::string st;
		ifs >> st;
		std::cout << st << "/";
	}
	std::cout << std::endl;
	*/
	return 0;
}
