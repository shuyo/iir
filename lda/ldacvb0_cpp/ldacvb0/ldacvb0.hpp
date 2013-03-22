/**
@file
@brief LDA CVB0

Copyright (C) 2013 Nakatani Shuyo / Cybozu Labs, Inc., all rights reserved.
*/

#ifndef LDACVB0_HPP
#define LDACVB0_HPP

#ifdef __linux__
#include <boost/regex.hpp>
namespace std {
    using boost::regex;
    using boost::match_results;
    using boost::regex_search;
}
#else
#include <regex>
#endif
#include <algorithm>
#include <string>
#include <unordered_map>
#include <cybozu/string.hpp>
#include <cybozu/string_operation.hpp>
#include <cybozu/mmap.hpp>

/*
 id and counter of vocabulary
*/
class idcount {
public:
	size_t id;
	int count;
	idcount() : id(0), count(0) {}
	idcount(size_t id_, int count_) : id(id_), count(count_) {}
};

/*
 vocabulary
*/
template <class KEY>
class vocabularies {
private:
	std::vector<KEY> vocalist;
	std::unordered_map<KEY, idcount> voca;

public:
	void operator()(const KEY &word) {
		KEY key(word);
		normalize(key);
		if (voca.find(key) != voca.end()) {
			voca[key].count += 1;
		} else {
			voca[key] = idcount(vocalist.size(), 1);
			vocalist.push_back(key);
		}
	}
	int size() const {
		return vocalist.size();
	}

	int count(const KEY &word) {
		return voca[word].count;
	}

	static void normalize(KEY &s) {
		cybozu::ToLower(s);
	}
};

std::regex rexword("[a-zA-Z]+");

/*
 call the procedure for each word (mmap)
 */
template <class T, class CHAR>
void eachwords(const CHAR* p, const CHAR* end, T& func) {
	std::regex_token_iterator<const CHAR*> i( p, end, rexword ), iend;
	while (i != iend) {
		func(*i++);
	}
};

/*
 call the procedure for each word (std::string, cybozu::String)
 */
template <class T, class STRING>
void eachwords(STRING st, T& func) {
	std::regex_token_iterator<STRING::iterator> i( st.begin(), st.end(), rexword ), iend;
	while (i != iend) {
		func(*i++);
	}
};

template <class T> void loadeachwords(std::string filename, T& func) {

	try {
		cybozu::Mmap map(filename);
		const char *p = map.get();
		const char *end = p + map.size();
		eachwords(p, end, func);

	} catch (std::exception& e) {
		printf("%s\n", e.what());
	}
}

class LDA_CVB0 {
private:
	int K_;
	double alpha_;
	double beta_;
public:
	LDA_CVB0(int K, double alpha, double beta) :
	  K_(K), alpha_(alpha), beta_(beta) {

	  }
};

#endif
