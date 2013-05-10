/**
@file
@brief LDA CVB0
Latent Dirichlet Allocation - Collapsed Variational Bayesian Estimation

Copyright (C) 2013 Nakatani Shuyo / Cybozu Labs, Inc., all rights reserved.
This code is licensed under the MIT license.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "cybozu/string.hpp"
#include <cybozu/mmap.hpp>
#include <cybozu/nlp/top_score.hpp>
#include "ldacvb0.hpp"

void printnwk(const cybozu::ldacvb0::LDA_CVB0& model, const std::string& word) {
	auto voca = model.docs_.vocabularies;
	size_t w = voca.id(word);
	auto i = model.n_wk.begin()+ w * model.K_;

	std::cout << "[" << word << "]" << std::endl;
	std::cout << "( ";
	for (size_t k=0;k<model.K_;++k) {
		std::cout << *(i+k) << " ";
	}
	std::cout << ")" << std::endl;
}

template <class STRING, class CHAR>
void printHighFreqWords(const cybozu::ldacvb0::Documents<STRING, CHAR> &docs) {
	for(auto df=docs.docfreq.begin(), dfend = docs.docfreq.end(); df!=dfend; ++df) {
		if (df->second > (int)M/2) {
			std::cout << docs.vocabularies.vocalist[df->first] << " " << df->second << std::endl;
		}
	}
}


int main(int argc, char* argv[]) {

	int K = 20, I = 100, N_WORDS = 20;
	size_t ldf = 1, udf = 0; // lower and upper limit of document frequency
	double alpha = 0.1;
	double beta = 0.01;
	bool isCorpusWithPos = false;

	std::vector<std::string> files;

	for(int i=1;i<argc;++i) {
		std::string st(argv[i]);

		if (st == "-k") {
			if (++i>=argc) goto ERROR_OPT_K;
			K = atoi(argv[i]);
		} else if (st == "-i") {
			if (++i>=argc) goto ERROR_OPT_I;
			I = atoi(argv[i]);
		} else if (st == "-n") {
			if (++i>=argc) goto ERROR_OPT_N;
			N_WORDS = atoi(argv[i]);
		} else if (st == "--ldf") {
			if (++i>=argc) goto ERROR_OPT_DF;
			ldf = atoi(argv[i]);
		} else if (st == "--udf") {
			if (++i>=argc) goto ERROR_OPT_DF;
			udf = atoi(argv[i]);
		} else if (st == "-a") {
			if (++i>=argc) goto ERROR_OPT_A;
			alpha = atof(argv[i]);
		} else if (st == "-b") {
			if (++i>=argc) goto ERROR_OPT_B;
			beta = atof(argv[i]);
		} else if (st == "-p") {
			isCorpusWithPos = true;
		} else {
			files.push_back(st);
		}
	}

	{
		cybozu::ldacvb0::Documents<std::string, char> orgdocs(isCorpusWithPos?cybozu::ldacvb0::REXWORD_WITH_POS:cybozu::ldacvb0::REXWORD), docs;

		for(auto i=files.begin(), iend=files.end();i!=iend;++i) {
			try {
				cybozu::Mmap map(*i);
				const char *p = map.get();
				const char *end = p + map.size();
				orgdocs.add(p, end);
			} catch (std::exception& e) {
				printf("%s\n", e.what());
			}
		}

		size_t M = orgdocs.size();
		size_t orgV = orgdocs.vocabularies.size();
		if (orgV <= 0) goto ERROR_NO_VOCA;

		if (udf == 0) udf = M / 2;
		truncDocFreq(docs, orgdocs, ldf, udf);

		size_t V = docs.vocabularies.size();
		if (V <= 0) goto ERROR_NO_VOCA;

		std::cout << "M = " << M;
		std::cout << ", N = " << docs.N;
		std::cout << ", V = " << V << " / " << orgV << std::endl;
		std::cout << "K = " << K << ", alpha = " << alpha << ", beta = " << beta << std::endl;

		cybozu::ldacvb0::LDA_CVB0 model(K, V, alpha, beta, docs);

		for(int i=0;i<I;++i) {
			std::cout << i << " " << model.perplexity() << std::endl;
			model.learn();
		}
		std::cout << "perplexity : " << model.perplexity() << std::endl;

		cybozu::ldacvb0::Vec worddist;
		model.worddist(worddist);
		auto voca = docs.vocabularies;
		for (int k=0;k<K;++k) {
			std::cout << std::endl << "[topic " << k << "]" << std::endl;

			cybozu::nlp::TopScore<size_t> ts(N_WORDS);
			size_t id = 0;
			for(auto i = worddist.begin() + k; id < V; i+=K, ++id) {
				ts.add(*i, id);
			}

			auto table = ts.getTable();
			auto tend = table.end();
			for (auto t = table.begin(); t!=tend; ++t) {
				const std::string& w = voca.vocalist[t->idx];
				std::cout << w << "\t" << docs.docfreq[t->idx] << "\t" << voca.count(w) << "\t" << t->score << std::endl;
			}
		}

		/*
		auto i = worddist.begin();
		//auto i = model.n_wk->begin();
		for(size_t id = 0; id < V;++id) {
			const std::string& w = voca.vocalist[id];
			std::cout << id << "\t" << w << "\t" << docs.docfreq[id] << "\t" << voca.count(w);
			for (int k=0;k<K;++k) {
				std::cout << "\t" << *i++;
			}
			std::cout << std::endl;
		}
		*/

		/*
		printnwk(model, "the");
		printnwk(model, "of");
		printnwk(model, "and");
		*/
	}




	return 0;



	/* error */

	char *p;
ERROR_OPT_K:
	p = "[ERROR] -k option needs positive integer";
	goto ERROR_EXIT;
ERROR_OPT_I:
	p = "[ERROR] -i option needs positive integer";
	goto ERROR_EXIT;
ERROR_OPT_N:
	p = "[ERROR] -n option needs positive integer";
	goto ERROR_EXIT;
ERROR_OPT_DF:
	p = "[ERROR] --ldf/udf option needs integer";
	goto ERROR_EXIT;
ERROR_OPT_A:
	p = "[ERROR] -a option needs positive real number";
	goto ERROR_EXIT;
ERROR_OPT_B:
	p = "[ERROR] -b option needs positive real number";
	goto ERROR_EXIT;
ERROR_NO_VOCA:
	p = "[ERROR] no vocabularies";
	goto ERROR_EXIT;

ERROR_EXIT:
	std::cerr << p << std::endl;
	return 1;

}
