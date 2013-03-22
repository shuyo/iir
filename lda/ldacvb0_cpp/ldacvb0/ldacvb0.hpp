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
#include <string>
#include <unordered_map>
#include <cybozu/string.hpp>
#include <cybozu/mmap.hpp>

class idcount {
public:
	size_t id;
	int count;
	idcount() : id(0), count(0) {}
	idcount(size_t id_, int count_) : id(id_), count(count_) {}
};

template <class String>
class vocabularies {
private:
	std::vector<String> vocalist;
	std::unordered_map<String, idcount > voca;

public:
	void operator()(const String &word) {
		//auto s = std::tolower
		if (voca.find(word) != voca.end()) {
			voca[word].count += 1;
		} else {
			voca[word] = idcount(vocalist.size(), 1);
			vocalist.push_back(word);
		}
	}
	int size() const {
		return vocalist.size();
	}
};

std::regex rexword("[a-zA-Z]+");

template <class T>
void eachwords(const char* p, const char* end, T& func) {
	std::regex_token_iterator<const char*> i( p, end, rexword ), iend;
	while (i != iend) {
		func(*i++);
	}
/*
	std::match_results<const char*> what;
	while (std::regex_search(p, end, what, r)) {
		std::string w(what[0].first, what[0].second);
		func(w.c_str());
		p = what[0].second;
	}*/
};

template <class T>
void eachwords(std::string st, T& func) {
	std::regex_token_iterator<std::string::iterator> i( st.begin(), st.end(), rexword ), iend;
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


	/*
	auto i = std::sregex_iterator(st.get(), st.get() + st.size(),  "[A-Za-z]+");
	auto iend = std::sregex_iterator();
	for (; i!=iend; ++i) {
		std::smatch m = *i;
		std::cout << m.str() << "/";
	}
	*/

/*
	//cybozu::String str(std::istreambuf_iterator<char>(ifs.rdbuf()), std::istreambuf_iterator<char>());
	while (ifs) {
		std::string st;
		ifs >> st;
		std::cout << st << "/";
	}
	*/

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
